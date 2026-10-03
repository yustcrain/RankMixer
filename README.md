### RankMixer

一种在精排阶段使用的混合排序方法。

#### 背景

由于普通的排序方法没有充分使用GPU的性能，且排序速度较慢，因此提出了这种方法。

#### method

RankMixer主要分为两个部分，第一部分为tokenmixing，第二部分为PFFN，考虑到不同特征直接到向量空间是异构的，因此tokenmixing阶段没有使用multhead self-attention，这在一定程度上节省了内存和时间开销。

假设输入为一个batch的向量，其中包含$T$个token，因此 $X=(x_{1}, x_{2}, \ldots, x_{T})$，同时假设每个token的维度为$d$，因此$X \in \mathbb{R}^{T \times d}$。在tokenmixing中，共有$d$个头，因此每个头处理的维度为$d_{head} = d / d_{head}$。

$$
S_{n-1}=layernorm(tokenmixing(X_{n-1})+X_{n-1}) 
$$
$$
X_{n}=layernorm(PFFN(S_{n-1})+S_{n-1})
$$

### Reference{
    https://arxiv.org/pdf/2507.15551v3
}