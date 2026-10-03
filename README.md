### RankMixer

一种在精排阶段使用的混合排序方法。

#### 背景

由于普通的排序方法没有充分使用GPU的性能，且排序速度较慢，因此提出了这种方法。

#### method

注：论文中的T和H是相等的，但是为了一般化，README中将T和H区分开来。

RankMixer主要分为两个部分，第一部分为tokenmixing，第二部分为PFFN，考虑到不同特征直接到向量空间是异构的，因此tokenmixing阶段没有使用multhead self-attention，同时这在一定程度上节省了内存和时间开销。

假设输入为一个batch的向量，其中包含$T$个token，因此 $X=(x_{1}, x_{2}, \ldots, x_{T})$，同时假设每个token的维度为$d$，因此$X \in \mathbb{R}^{T \times d}$。在tokenmixing中，共有$d$个头，因此每个头处理的维度为$d_{head} = d / d_{head}$。

$$
S_{n-1}=layernorm(tokenmixing(X_{n-1})+X_{n-1}) 
$$
$$
X_{n}=layernorm(PFFN(S_{n-1})+S_{n-1})
$$

#### tokenmixing

tokenmixing部分主要是将token向量进行分块并进行重新拼接，具体的操作为：
- (B, T, D) -> (B, T, H, D/H) -> (B, H, T, D/H) -> (B, H, T $\times$ D/H)

#### PFFN

每个头的token会进入FFN进行处理，维度为(B, T $\times$ D/H)，经过FFN后，维度不变。这里有两条路径，其中一条是进入router进行路由，另一条是直接进入FFN进行处理，最后将两条路径的结果进行加权求和。


### Reference{
    https://arxiv.org/pdf/2507.15551v3
}