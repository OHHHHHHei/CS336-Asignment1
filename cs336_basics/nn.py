import torch as torch
import torch.nn as nn

# 定义线性层，继承自 nn.Module
# 线性层的作用是对输入进行线性变换，通常表示为 y = xW^T + b，其中 W 是权重矩阵，b 是偏置向量。
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features

        # 因为输入的一般是一个(batch_size, in_features)的张量，而输出是(batch_size, out_features)
        # 所以权重矩阵的维度应该是(out_features, in_features)，这样在进行矩阵乘法时才能得到正确的输出维度。
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # 为了使得权重的初始值不太大，我们使用了一个基于输入和输出特征数量的标准差来初始化权重。
        # 这就是Xavier初始化，核心作用就是防止多层网络中方差消失或爆炸的问题。
        std = (2.0 / (in_features + out_features)) ** 0.5

        # trunc_normal_函数会生成一个截断的正态分布随机数，mean是均值，std是标准差，a和b分别是截断的下界和上界。
        # 这里我们将权重初始化为一个均值为0，标准差为std，并且在-3*std到3*std之间的截断正态分布随机数。
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        # 直接进行无 bias 的线性变换
        # return x @ self.weight.T

        # 使用 einsum 函数，来适应各种 batch 维度情况
        # einsum 的字符串参数 '...i, oi -> ...o' 表示：
        # - '...i' 表示输入张量 x 的最后一个维度是 i，前面的维度可以是任意的（例如 batch_size）。
        # - 'oi' 表示权重矩阵 self.weight 的维度是 (out_features, in_features)
        # 其中 o 是输出特征的维度，i 是输入特征的维度。
        return torch.einsum('...i, oi -> ...o', x, self.weight)

# 定义 embedding 类，继承自 nn.Module
# 本质上就是一个 token id 到 embedding 向量的映射
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim: int, device = None, dtype = None):
        # 调用 nn.Module 的初始化函数
        super().__init__()

        # 方便后续创建参数时指定设备和数据类型
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 创建一个可训练的参数矩阵，大小为 (vocab_size, embedding_dim)，用于存储词嵌入， empty() 函数创建一个未初始化的张量
        # 只有被 nn.Parameter 包起来的 tensor，PyTorch 才会把它当作模型参数，训练的时候 optimizer 才会更新它。
        self.weight = nn.Parameter(torch.empty((vocab_size, embedding_dim), **factory_kwargs))
        # 初始化 embedding 矩阵
        # trunc_normal_ 是截断正态分布初始化
        # mean 是均值，std 是标准差，a 和 b 是截断范围的下界和上界
        # 最后的下划线表示这是一个原地操作，会直接修改 self.weight 的值
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]