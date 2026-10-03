import torch
from torch import nn

from rank_mixer.layers import MHTokenMixingBlock, PFFN, PReMoE


class RankMixer(nn.Module):
    """
    RankMixer: Full ranking model with stacked RankMixerBlocks and prediction head.

    Architecture:
        Input (B, T, D)
        -> RankMixerBlock x num_layers
        -> Prediction Head (Linear)
        -> Output (B, d_out)

    Args:
        d_model: Model dimension (D)
        token_dim: Number of tokens (T)
        num_heads: Number of attention heads
        num_layers: Number of RankMixerBlock layers to stack
        d_out: Output dimension (e.g., 1 for binary classification)
        per_token_type: Type of per-token layer ("pffn" or "premoe")
        ffn_expansion_ratio: Expansion ratio for FFN/MoE intermediate dimension
        num_experts: Number of experts (only used if per_token_type="premoe")
        eps: LayerNorm epsilon
    """

    def __init__(
        self,
        d_model: int,
        token_dim: int,
        num_heads: int,
        num_layers: int = 2,
        d_out: int = 1,
        per_token_type: str = 'pffn',
        ffn_expansion_ratio: int = 4,
        num_experts: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.d_model = d_model
        self.token_dim = token_dim
        self.num_layers = num_layers
        self.ffn_expansion_ratio = ffn_expansion_ratio
        self.num_experts = num_experts
        self.per_token_type = per_token_type
        self.d_out = d_out

        self.blocks = nn.ModuleList(
            [
                RankMixerBlock(
                    d_model=d_model,
                    token_dim=token_dim,
                    num_heads=num_heads,
                    per_token_type=per_token_type,
                    ffn_expansion_ratio=ffn_expansion_ratio,
                    num_experts=num_experts,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.head = nn.Linear(d_model * token_dim, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f'Expected input of shape (B, T, D), got {x.shape}')

        B, T, D = x.shape
        if self.token_dim != T or self.d_model != D:
            raise ValueError(f'Input shape mismatch: expected (*, {self.token_dim}, {self.d_model}), got {x.shape}')

        for block in self.blocks:
            x = block(x)

        x = x.reshape(B, -1)
        x = self.head(x)

        return x


class RankMixerBlock(nn.Module):
    """
    Full RankMixer Block with configurable per-token sublayer.

    Structure (Post-Norm):

        s = LN(TokenMixing(x) + x)
        out = LN(PerTokenLayer(s) + s)

    Input / Output shape:
        (B, T, D)

    Notes:
        - TokenMixing is parameter-free.
        - Per-token layer can be PFFN or PReMoE.
        - LayerNorm is applied after each residual addition.
    """

    def __init__(
        self,
        d_model: int,
        token_dim: int,
        num_heads: int,
        per_token_type: str = 'pffn',
        ffn_expansion_ratio: int = 4,
        num_experts: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()

        assert per_token_type in {'pffn', 'premoe'}, "per_token_type must be either 'pffn' or 'premoe'"

        self.d_model = d_model
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.ffn_expansion_ratio = ffn_expansion_ratio
        self.per_token_type = per_token_type
        self.num_experts = num_experts if per_token_type == 'premoe' else None

        self.token_mixer = MHTokenMixingBlock(
            d_model=d_model,
            num_heads=num_heads,
            token_dim=token_dim,
        )
        self.ln_token = nn.LayerNorm(d_model, eps=eps)

        if per_token_type == 'pffn':
            self.per_token_layer = PFFN(
                d_model=d_model,
                token_dim=token_dim,
                expansion_ratio=ffn_expansion_ratio,
            )
        else:
            self.per_token_layer = PReMoE(
                d_model=d_model,
                token_dim=token_dim,
                num_experts=num_experts,
                expansion_ratio=ffn_expansion_ratio,
            )

        self.ln_per_token = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f'Expected input of shape (B, T, D), got {x.shape}')

        s_n = self.ln_token(self.token_mixer(x)+x)
        x_n = self.ln_per_token(self.per_token_layer(s_n)+s_n)
        return x_n


def base(
    d_model: int = 128,
    token_dim: int = 16,
    num_layers: int = 2,
    d_out: int = 1,
    ffn_expansion_ratio: int = 4,
    eps: float = 1e-5,
) -> RankMixer:
    return RankMixer(
        d_model=d_model,
        token_dim=token_dim,
        num_heads=token_dim,
        num_layers=num_layers,
        d_out=d_out,
        per_token_type='pffn',
        ffn_expansion_ratio=ffn_expansion_ratio,
        num_experts=4,
        eps=eps,
    )


def moe_small(
    d_model: int = 128,
    token_dim: int = 16,
    num_layers: int = 2,
    d_out: int = 1,
    ffn_expansion_ratio: int = 4,
    num_experts: int = 4,
    eps: float = 1e-5,
) -> RankMixer:
    return RankMixer(
        d_model=d_model,
        token_dim=token_dim,
        num_heads=token_dim,
        num_layers=num_layers,
        d_out=d_out,
        per_token_type='premoe',
        ffn_expansion_ratio=ffn_expansion_ratio,
        num_experts=num_experts,
        eps=eps,
    )


def moe_large(
    d_model: int = 192,
    token_dim: int = 24,
    num_layers: int = 3,
    d_out: int = 1,
    ffn_expansion_ratio: int = 4,
    num_experts: int = 8,
    eps: float = 1e-5,
) -> RankMixer:
    return RankMixer(
        d_model=d_model,
        token_dim=token_dim,
        num_heads=token_dim,
        num_layers=num_layers,
        d_out=d_out,
        per_token_type='premoe',
        ffn_expansion_ratio=ffn_expansion_ratio,
        num_experts=num_experts,
        eps=eps,
    )