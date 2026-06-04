import torch
from cs336_basics.nn import softmax

# 概率过滤，只保留概率大于设定值的 token
def apply_top_p(probs:torch.Tensor, top_p:float) -> torch.Tensor:

    # 按照最后一维，把概率从大到小排序。最后一维就是 vocab_size
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # 计算累积概率，计算当前 token 之前的概率之和。
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 创建布尔 mask，表示那些 token 要被保留
    keep_mask = cumulative_probs <= top_p

    # 确保至少保留了一个 token，即使它的概率超过了 top_p。
    # 因为如果 top_p 非常小，可能会导致所有 token 都被过滤掉。
    keep_mask[..., 0] = True

    # 把不保留的 token 概率设为 0
    filtered_sorted_probs = sorted_probs * keep_mask

    # 创建一个和原始 probs 形状一样的 tensor
    filtered_probs = torch.zeros_like(probs)

    # 把过滤后的概率放回原来的位置。
    # sorted_indices 告诉我们每个 token 在原始 probs 中的位置。
    # 恢复成原始词表顺序
    filtered_probs.scatter_(
        dim=-1,
        index=sorted_indices,
        src=filtered_sorted_probs,
    )

    # 归一化概率，使它们的和为 1。因为我们把一些 token 的概率设为 0，所以需要重新归一化。
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    # 返回过滤后的概率分布
    return filtered_probs


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    context_length,
    temperature=1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    device=None,
) -> str:
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    # 将 prompt 编码成 token
    token_ids = tokenizer.encode(prompt)
    # 转变成 tensor
    generated = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 取出当前生成的序列的最后 context_length 个 token 作为输入
            input_ids = generated[:, -context_length:]

            # 前向传播，得到下一个 token 的 logits
            logits = model(input_ids)

            # 取出最后一个 token 的 logits
            next_token_logits = logits[:, -1, :]

            # 根据 temperature 调整 logits 的分布
            if temperature == 0:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature

                # 计算概率分布
                probs = softmax(next_token_logits, dim=-1)

                # 如果设置了 top_p，则应用概率过滤
                if top_p is not None:
                    probs = apply_top_p(probs, top_p)

                # 从概率分布中采样下一个 token 的 id
                next_token_id = torch.multinomial(probs, num_samples=1)

            # 将采样得到的 token id 添加到生成序列中
            generated = torch.cat((generated, next_token_id), dim=1)

            # 如果采样到 EOS token，则停止生成
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
    # 返回生成的文本，使用 tokenizer 解码生成的 token id 序列。
    # generated[0] 是因为 generated 的形状是 (1, seq_len)，我们只需要第一行的 token id 序列。
    return tokenizer.decode(generated[0].tolist())

    
    