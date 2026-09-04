import argparse
import json
from pathlib import Path

import torch

from cs336_basics.generate import generate
from cs336_basics.nn import TransformerLM
from cs336_basics.tokenizer import load_tokenizer_files

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a TinyStories checkpoint.")
    parser.add_argument("--prompt", default="Hello, she said")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tinystories_base/generated_sample.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    prompt = args.prompt

    # 生成回复
    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=device,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(text + "\n", encoding="utf-8")

    token_count = len(tokenizer.encode(text))
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint["iteration"],
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "token_count_including_prompt": token_count,
        "output_path": str(args.output_path),
    }
    metadata_path = args.output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(text)
    print(f"Saved generated text to {args.output_path}")
    print(f"Saved generation metadata to {metadata_path}")

if __name__ == "__main__":
    main()
