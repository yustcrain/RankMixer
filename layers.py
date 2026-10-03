import torch
import torch.nn.functional as F
from torch import nn


class MHTokenMixingBlock(nn.Module):
    """
    Multi-Head Token Mixing block (RankMixer).

    Input shape:
        x: (B, T, D)

    Output shape:
        out: (B, T, D)

    Assumptions (paper-faithful):
        - num_heads == token_dim == T
        - d_model % num_heads == 0

    This block is parameter-free and must be used with
    residual connection + LayerNorm by the caller.
    """

    def __init__(self, d_model: int, num_heads: int, token_dim: int):
        super().__init__()

        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        assert token_dim == num_heads, 'RankMixer TokenMixing requires num_heads == token_dim (H == T)'

        self.d_model = d_model  # D
        self.num_heads = num_heads  # H
        self.token_dim = token_dim  # T
        self.d_h = d_model // num_heads  # D // H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, D)

        Returns:
            Tensor of shape (B, T, D)
        """
        if x.dim() != 3:
            raise ValueError(f'Expected input of shape (B, T, D), got {x.shape}')

        B, T, D = x.shape
        if self.token_dim != T or self.d_model != D:
            raise ValueError(f'Input shape mismatch: expected (*, {self.token_dim}, {self.d_model}), got {x.shape}')

        x = x.reshape(B, T, self.num_heads, self.d_h)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, self.num_heads, T * self.d_h)
        out = x
        return out


class FFN(nn.Module):
    """
    Standard position-wise Feed-Forward Network.

    Applies the same FFN to every token.
    Input/Output shape:
        (B, T, D) -> (B, T, D)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PFFN(nn.Module):
    """
    Per-Token Feed-Forward Network (RankMixer).

    Each token has its *own* FFN parameters.

    Input shape:
        x: (B, T, D)

    Output shape:
        out: (B, T, D)

    Notes:
        - No parameters are shared across tokens
        - Must be wrapped with residual + LayerNorm by the caller
    """

    def __init__(self, d_model: int, token_dim: int, expansion_ratio: int = 4):
        super().__init__()

        self.d_model = d_model
        self.token_dim = token_dim
        self.d_ff = d_model * expansion_ratio

        # One FFN per token (no parameter sharing)
        self.ffns = nn.ModuleList([FFN(d_model, self.d_ff) for _ in range(token_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f'Expected input of shape (B, T, D), got {x.shape}')

        _, T, D = x.shape
        if self.token_dim != T or self.d_model != D:
            raise ValueError(f'Input shape mismatch: expected (*, {self.token_dim}, {self.d_model}), got {x.shape}')

        outputs = []
        for t in range(self.token_dim):
            outputs.append(self.ffns[t](x[:, t, :]))

        out = torch.stack(outputs, dim=1)
        return out


class ReLURouter(nn.Module):
    """
    ReLU-based routing gate.

    Computes non-negative routing weights without Top-K or Softmax.

    Input:
        x: (B, D)

    Output:
        gates: (B, E) non-negative
    """

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))


class ReMoE(nn.Module):
    """
    ReLU-based Mixture-of-Experts block.

    Key properties:
        - No Top-K routing
        - Sparse activation via ReLU
        - Fully differentiable
        - Token-adaptive expert counts

    Input / Output:
        x: (B, D)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expansion_ratio: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.d_ff = d_model * expansion_ratio

        self.router = ReLURouter(d_model, num_experts)
        self.experts = nn.ModuleList([FFN(d_model, self.d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, D)

        Returns:
            Tensor of shape (B, D)
        """
        if x.dim() != 2:
            raise ValueError(f'Expected input of shape (B, D), got {x.shape}')

        _, D = x.shape
        if self.d_model != D:
            raise ValueError(f'Input shape mismatch: expected (*, {self.d_model}), got {x.shape}')

        gates = self.router(x)

        expert_outputs = []
        for e in range(self.num_experts):
            out_e = self.experts[e](x)
            expert_outputs.append(out_e)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        out = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)

        return out


class PReMoE(nn.Module):
    """
    Per-Token ReMoE block (RankMixer).

    Each token has its own ReMoE (no parameter sharing across tokens).

    Input / Output:
        (B, T, D) -> (B, T, D)
    """

    def __init__(
        self,
        d_model: int,
        token_dim: int,
        num_experts: int,
        expansion_ratio: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.token_dim = token_dim

        self.remoes = nn.ModuleList([ReMoE(d_model, num_experts, expansion_ratio) for _ in range(token_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f'Expected input of shape (B, T, D), got {x.shape}')

        _, T, D = x.shape
        if self.token_dim != T or self.d_model != D:
            raise ValueError(f'Input shape mismatch: expected (*, {self.token_dim}, {self.d_model}), got {x.shape}')

        outputs = []
        for t in range(self.token_dim):
            outputs.append(self.remoes[t](x[:, t, :]))

        return torch.stack(outputs, dim=1)