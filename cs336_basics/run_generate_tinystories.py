from pathlib import Path

import torch

from cs336_basics.generate import generate
from cs336_basics.nn import TransformerLM
from cs336_basics.tokenizer import load_tokenizer_files

def main() -> None:

    # 分词器路径
    tokenizer_dir = Path("/data/leejt/cs336_assignment1/data/TinyStoriesV2-GPT4-train")
    checkpoint_path = Path("/data/leejt/cs336_assignment1/runs/tinystories_base/checkpoint.pt")

    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    special_tokens = ["<|endoftext|>"]

    # 加载分词器
    tokenizer = load_tokenizer_files(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=special_tokens,
    )
    # 拿到 eos token
    eos_token_id = tokenizer.byte_to_id["<|endoftext|>".encode("utf-8")]

    # 初始化模型配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 10000
    context_length = 256

    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    theta = 10000.0

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=num_heads,
        d_ff=d_ff,
        context_length=context_length,
        theta=theta,
        device=device,
    )

    # 加载 ckpt
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")

    # prompt 内容
    prompt = "Hello, she said"

    # 生成回复
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=200,
        context_length=context_length,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=eos_token_id,
        device=device,
    )

    print(text)

if __name__ == "__main__":
    main()
