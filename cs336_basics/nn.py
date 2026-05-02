import torch as torch
import torch.nn as nn
from einops import rearrange


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
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 必须初始化为 1，否则在训练过程中会出现数值不稳定的问题
        # 因为 RMSNorm 的计算涉及到除以输入的均方根，如果权重初始化为 0，可能会导致除以零的情况。

        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x):
        # 计算输入 x 的均方根，保持维度不变以便后续广播
        x_float= x.to(torch.float32)  # 确保计算均方根时使用浮点数

        """
        在 PyTorch（以及 NumPy）中，广播（Broadcasting） 是指在对两个形状不同的张量进行算术运算时，系统自动“扩展”较小张量的维度，使其与较大张量匹配的机制。
        要使两个张量是可广播的（Broadcastable），必须满足以下核心规则：
        核心规则：从右往左看
            比较两个张量的形状时，要从最后一个维度（最右边）开始往前检查。对于每一对对应的维度，必须满足以下 两个条件之一：
                1.这两个维度的值相等。
                2.其中一个维度的值是 1。
            如果其中一个张量的维度较少，系统会自动在它的左侧补 1，直到两者的维度数量相等，然后再按上述规则检查。
        """

        # -1 表示在最后一个维度上计算均值，keepdim=True 保持维度不变以便后续广播
        ms = x_float.pow(2).mean(-1, keepdim=True)  # 计算均方
        rms = torch.sqrt(ms + self.eps)  # 加上一个小的常数以避免除以零

        result = x_float / rms * self.weight  # 对输入进行归一化，并乘以权重
        # 对输入进行归一化，并乘以权重
        return result.to(x.dtype)  # 将结果转换回输入的原始数据类型

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    # d_ff 是前馈网络的隐藏层维度，d_model 是输入和输出的维度
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_ff = d_ff
        self.d_model = d_model

        # SwiGLU 的计算涉及到两个线性变换：一个用于计算 gate（门控信号），另一个用于计算 signal（信号）
        # 最后的输出是 gate 和 signal 的逐元素乘积，再经过一个线性变换得到最终的输出。
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x):

        gate = SiLU(self.w1(x))
        signal = self.w3(x)

        return self.w2(gate * signal)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta, d_k, context_length, device=None):
        """
        初始化 RoPE 模块
        theta: 基准频率 (通常为 10000)
        d_k: 每个 Head 的维度 (必须是偶数)
        context_length: 最大序列长度
        """
        super().__init__()
        # d_k 是每个 Head 的维度，必须是偶数，因为 RoPE 需要将维度分成两部分来计算旋转位置编码。
        self.d_k = d_k
        # 计算频率 
        # powers 是一个从 0 到 d_k/2-1 的整数序列，每个元素除以 d_k 得到一个归一化的指数值
        # 这些指数值用于计算不同频率的正弦和余弦函数。
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)  # 计算频率，得到一个长度为 d_k/2 的频率向量

        # 标记 token 的位置，t 是一个从 0 到 context_length-1 的整数序列
        # 表示每个 token 在序列中的位置。
        t = torch.arange(context_length, device=device).float()
        # 计算频率矩阵，通过外积（outer product）将位置向量 t 和频率向量 freqs 结合起来
        # 得到一个形状为 (context_length, d_k/2) 的矩阵。
        frqs_matrix = torch.outer(t, freqs)

        self.register_buffer('cos_cached', torch.cos(frqs_matrix))
        self.register_buffer('sin_cached', torch.sin(frqs_matrix))

    def forward(self, x, token_positions):
        """
        x: 输入张量，形状为 (batch_size, seq_len, d_k) 是要被 RoPE 旋转的 query 或 key。
        token_positions: 每个 token 的位置索引，形状为 (batch_size, seq_len)
        """
        # 从缓存中获取对应位置的 cos 和 sin 值
        cos = self.cos_cached[token_positions]  # 形状为 (batch_size, seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # 形状为 (batch_size, seq_len, d_k/2)

        if x.ndim > cos.ndim:
            # 如果输入张量 x 的维度比 cos 和 sin 的维度多，说明 x 可能有一个额外的 batch 维度
            # 这种情况下，我们需要在 cos 和 sin 的前面添加一个新的维度，以便它们能够正确地广播到 x 的形状。
            cos = cos.unsqueeze(0)  # 在第0维添加一个新的维度，变成 (1, batch_size, seq_len, d_k/2)
            sin = sin.unsqueeze(0)  # 同上
        
        cos = cos.to(x.dtype)  # 确保 cos 和 sin 的数据类型与输入 x 一致
        sin = sin.to(x.dtype)

        x_even = x[..., ::2]  # 取出 x 的偶数索引维度，形状为 (batch_size, seq_len, d_k/2)
        x_odd = x[..., 1::2]  # 取出 x 的奇数索引维度，形状为 (batch_size, seq_len, d_k/2)

        # 应用 RoPE 公式进行旋转位置编码
        x_rotated_1 = x_even * cos - x_odd * sin
        x_rotated_2 = x_even * sin + x_odd * cos

        output = torch.empty_like(x)  # 创建一个与输入 x 形状相同的空张量，用于存储旋转后的结果
        output[..., ::2] = x_rotated_1  # 将旋转后的偶数索引维度放回输出张量的偶数索引位置
        output[..., 1::2] = x_rotated_2  # 将旋转后的奇数索引维度放回输出张量的奇数索引位置

        return output

def softmax(x, dim):
    # 为了数值稳定性，在计算 softmax 之前，我们通常会减去输入张量 x 的最大值
    # 这样可以防止在计算指数函数时出现数值溢出的问题。
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)

    output = exp_x / sum_exp_x
    return output

# 计算缩放点积注意力的函数
def scaled_dot_product_attention(
    Q:torch.Tensor, 
    K:torch.Tensor, 
    V:torch.Tensor, 
    mask:torch.Tensor=None
):
    # 取 Q 的最后一维大小
    d_k = Q.shape[-1]
    
    score = torch.einsum('...qd,...kd->...qk', Q, K) / (d_k ** 0.5)
    # 如果有 mask，就掩码
    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))

    probs = softmax(score, dim=-1)

    # 乘以 V 得到最后的输出
    output = torch.einsum('...qk,...kv->...qv', probs, V)
    return output


class CasualSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        context_length=None,
        theta=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads

        # 定义线性层，用于计算 Q、K、V
        self.w_q = Linear(d_model, d_model, **factory_kwargs)
        self.w_k = Linear(d_model, d_model, **factory_kwargs)
        self.w_v = Linear(d_model, d_model, **factory_kwargs)

        # 输出线性层
        self.w_o = Linear(d_model, d_model, **factory_kwargs)

        if theta is not None and context_length is not None:
            self.rope = RotaryPositionEmbedding(theta, self.d_k, context_length, device=device)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        # 计算 Q、K、V，并将它们的形状调整为 (batch_size, n_heads, seq_len, d_k)
        q = rearrange(self.w_q(x), 'b s (h d) -> b h s d', h=self.n_heads)
        k = rearrange(self.w_k(x), 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(self.w_v(x), 'b s (h d) -> b h s d', h=self.n_heads)

        if self.rope is not None:
            if token_positions is None:
                # 适配 token_position 的形状，确保它与 q、k 的 batch 维度和序列长度匹配
                batch_dims = x.shape[:-2]
                token_positions = torch.arange(x.shape[-2], device=x.device).expand(*batch_dims, -1)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], device=x.device)).bool()
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')

        output = self.w_o(attn_output)
        return output

class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int,
                 theta: float, device=None, dtype=None, 
                 use_rms_norm: bool = True,
                 norm_mode: str = "pre",   # 选项: "pre", "post"
                 ffn_type: str = "swiglu"  # 选项: "swiglu", "silu"
                 ):
            super().__init__()
            self.use_rms_norm = use_rms_norm
            self.norm_mode = norm_mode
            self.ffn_type = ffn_type

            factory_kwargs = {'device': device, 'dtype': dtype}
            self.attn = CasualSelfAttention(d_model, num_heads, context_length=context_length, theta=theta, **factory_kwargs)

            # 初始化 Norm 层
            if use_rms_norm:
                self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
                self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
            else:
                self.norm1 = nn.Identity()
                self.norm2 = nn.Identity()


            # 初始化 FFN 层
            if ffn_type == "swiglu":
                self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            elif ffn_type == "silu":
                d_ff = 4 * d_model  # 通常情况下，SILU 的 FFN 隐藏层维度是输入维度的4倍
                self.ffn = nn.Sequential(
                    Linear(d_model, d_ff, device=device, dtype=dtype),
                    nn.SiLU(),
                    Linear(d_ff, d_model, device=device, dtype=dtype)
                )
            else:
                raise ValueError(f"Unsupported ffn_type: {ffn_type}")

        def forward(self, x, token_positions=None):
            # 根据 norm_mode 决定是 Pre-Norm 还是 Post-Norm
            if self.norm_mode == "pre":
                # Pre-Norm: 先归一化，再计算注意力和 FFN
                x_norm = self.norm1(x)
                attn_output = self.attn(x_norm, token_positions=token_positions)
                x = x + attn_output  # 残差连接

                x_norm = self.norm2(x)
                ffn_output = self.ffn(x_norm)
                x = x + ffn_output  # 残差连接
            elif self.norm_mode == "post":
                # Post-Norm: 先计算注意力和 FFN，再归一化
                attn_output = self.attn(x, token_positions=token_positions)
                x = x + attn_output  # 残差连接
                x = self.norm1(x)

                ffn_output = self.ffn(x)
                x = x + ffn_output  # 残差连接
                x = self.norm2(x)
            else:
                raise ValueError(f"Unsupported norm_mode: {self.norm_mode}")

            return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model,num_layers, n_heads, d_ff, context_length, theta, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.token_embedding = Embedding(vocab_size, d_model, **factory_kwargs)


        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=n_heads,
                d_ff=d_ff,
                context_length=context_length,
                theta=theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        # 最后的输出层之前的归一化层
        # 通常在 Transformer 模型中会在输出层之前添加一个归一化层，以帮助稳定训练过程并提高模型性能。
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # 输出层，将 Transformer 的输出映射到词汇表大小的维度
        # 以便进行语言建模任务中的下一个 token 预测。
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)

        # 生成 token 位置索引，形状为 (batch_size, seq_len)
        # 其中每个元素表示对应 token 在序列中的位置。
        token_positions = torch.arange(token_ids.shape[-1], device=token_ids.device).expand(token_ids.shape[0], -1) 
        # 将 token 位置索引传递给每个 TransformerBlock
        # 以便在计算注意力时使用 RoPE 进行位置编码。
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        # 在所有 TransformerBlock 处理完后，先进行一次归一化
        # 然后通过 lm_head 线性层得到最终的 logits 输出。
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
    






