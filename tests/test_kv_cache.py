import torch

from cs336_basics.generate import generate
from cs336_basics.nn import TransformerLM


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=32,
        d_model=16,
        num_layers=2,
        n_heads=4,
        d_ff=32,
        context_length=16,
        theta=10000.0,
    )
    model.eval()

    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    full_logits = model(token_ids)

    # 先处理 prompt，再逐个 token 使用 cache。
    prompt_logits, cache = model(token_ids[:, :3], use_cache=True)
    k_storage = cache[0].k
    v_storage = cache[0].v
    token4_logits, cache = model(
        token_ids[:, 3:4],
        past_key_values=cache,
        use_cache=True,
    )
    token5_logits, _ = model(
        token_ids[:, 4:5],
        past_key_values=cache,
        use_cache=True,
    )

    cached_logits = torch.cat(
        [prompt_logits, token4_logits, token5_logits],
        dim=1,
    )
    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-5)
    assert cache[0].k.data_ptr() == k_storage.data_ptr()
    assert cache[0].v.data_ptr() == v_storage.data_ptr()
    assert cache[0].valid_length == 5


class ToyTokenizer:
    def encode(self, text):
        return [1, 2]

    def decode(self, token_ids):
        return ",".join(str(token_id) for token_id in token_ids)


def test_cached_generation_rebuilds_full_context_window():
    torch.manual_seed(1)
    model = TransformerLM(
        vocab_size=32,
        d_model=16,
        num_layers=2,
        n_heads=4,
        d_ff=32,
        context_length=4,
        theta=10000.0,
    )
    model.eval()

    tokenizer = ToyTokenizer()
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="prompt",
        max_new_tokens=8,
        context_length=4,
        temperature=0,
        top_p=None,
        device=torch.device("cpu"),
    )
    full_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="prompt",
        max_new_tokens=8,
        context_length=4,
        temperature=0,
        top_p=None,
        device=torch.device("cpu"),
        use_cache=False,
    )

    generated = torch.tensor([[1, 2]])
    for _ in range(8):
        logits = model(generated[:, -4:])
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    assert output == tokenizer.decode(generated[0].tolist())
    assert output == full_output
