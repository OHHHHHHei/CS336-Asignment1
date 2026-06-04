import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import load_tokenizer_files


def encode_file(input_path: Path, output_path: Path, tokenizer, dtype: np.dtype) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_tokens = 0

    with open(input_path, "r", encoding="utf-8") as input_file, open(output_path, "wb") as output_file:
        for line in tqdm(input_file, desc=f"Encoding {input_path.name}"):
            token_ids = tokenizer.encode(line)
            token_array = np.asarray(token_ids, dtype=dtype)
            token_array.tofile(output_file)
            num_tokens += token_array.size

    return num_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode TinyStories text files with a trained BPE tokenizer.")
    parser.add_argument("--data-dir", type=Path, default=Path("/data/leejt/cs336_assignment1/data"))
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    data_dir = args.data_dir
    tokenizer_dir = args.tokenizer_dir or data_dir / "TinyStoriesV2-GPT4-train"
    output_dir = args.output_dir or data_dir / "TinyStoriesV2-GPT4-tokenized"

    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    tokenizer = load_tokenizer_files(vocab_path, merges_path, special_tokens=args.special_token)
    dtype = np.uint16 if len(tokenizer.vocab) <= np.iinfo(np.uint16).max else np.uint32

    files = {
        "train": data_dir / "TinyStoriesV2-GPT4-train.txt",
        "valid": data_dir / "TinyStoriesV2-GPT4-valid.txt",
    }

    metadata = {
        "tokenizer_dir": str(tokenizer_dir),
        "dtype": np.dtype(dtype).name,
        "vocab_size": len(tokenizer.vocab),
        "files": {},
    }

    start_time = time.time()
    for split, input_path in files.items():
        output_path = output_dir / f"{split}.bin"
        num_tokens = encode_file(input_path, output_path, tokenizer, dtype)
        metadata["files"][split] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "num_tokens": num_tokens,
        }

    metadata["elapsed_seconds"] = time.time() - start_time
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved tokenized data to {output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
