import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile TinyStories BPE training without saving a tokenizer.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--profile-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    profiler = cProfile.Profile() if args.profile_path else None
    if profiler is not None:
        profiler.enable()

    vocab, merges = train_bpe(args.input_path, args.vocab_size, special_tokens)

    if profiler is not None:
        profiler.disable()
        args.profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile_path)

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
        stats.print_stats(40)
        if args.summary_path is not None:
            args.summary_path.parent.mkdir(parents=True, exist_ok=True)
            args.summary_path.write_text(stream.getvalue(), encoding="utf-8")

    summary = {
        "input_path": str(args.input_path),
        "input_bytes": args.input_path.stat().st_size,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
