import argparse
import json
from pathlib import Path
import numpy as np
import torch

from cs336_basics.data import load_tokenized_data
from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.train import train
from cs336_basics.logging_utils import save_config


EXPERIMENTS = {
    "baseline": {
        "run_name": "tinystories_base",
        "use_rms_norm": True,
        "norm_mode": "pre",
        "use_rope": True,
        "ffn_type": "swiglu",
    },
    "no_rmsnorm": {
        "run_name": "tinystories_no_rmsnorm",
        "use_rms_norm": False,
        "norm_mode": "pre",
        "use_rope": True,
        "ffn_type": "swiglu",
    },
    "post_norm": {
        "run_name": "tinystories_post_norm",
        "use_rms_norm": True,
        "norm_mode": "post",
        "use_rope": True,
        "ffn_type": "swiglu",
    },
    "nope": {
        "run_name": "tinystories_nope",
        "use_rms_norm": True,
        "norm_mode": "pre",
        "use_rope": False,
        "ffn_type": "swiglu",
    },
    "silu": {
        "run_name": "tinystories_silu",
        "use_rms_norm": True,
        "norm_mode": "pre",
        "use_rope": True,
        "ffn_type": "silu",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TinyStories Transformer ablation.")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("/data/leejt/cs336_assignment1/data/TinyStoriesV2-GPT4-tokenized"))
    parser.add_argument("--run-root", type=Path, default=Path("/data/leejt/cs336_assignment1/runs"))
    parser.add_argument("--num-iters", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--num-eval-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--cosine-cycle-iters", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    experiment = EXPERIMENTS[args.experiment]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 数据加载部分
    data_dir = args.data_dir
    train_path = data_dir / "train.bin"
    valid_path = data_dir / "valid.bin"
    train_data = load_tokenized_data(str(train_path), dtype=np.uint16)
    valid_data = load_tokenized_data(str(valid_path), dtype=np.uint16)

    # 模型配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 10000
    context_length = 256

    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    theta = 10000.0

    batch_size = args.batch_size
    num_iters = args.num_iters
    eval_interval = args.eval_interval
    num_eval_batches = args.num_eval_batches

    max_l2_norm = 1.0
    max_learning_rate = args.max_learning_rate
    min_learning_rate = args.min_learning_rate
    warmup_iters = args.warmup_iters
    cosine_cycle_iters = args.cosine_cycle_iters or num_iters

    # 模型、优化器和训练循环
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        device=device,
        use_rms_norm=experiment["use_rms_norm"],
        norm_mode=experiment["norm_mode"],
        use_rope=experiment["use_rope"],
        ffn_type=experiment["ffn_type"],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    run_name = args.run_name or experiment["run_name"]
    run_dir = args.run_root / run_name
    # 确保运行目录存在
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    # 检查点路径
    checkpoint_path = run_dir / "checkpoint.pt"

    config = {
        "experiment": args.experiment,
        "run_name": run_name,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "device": str(device),
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "theta": theta,
        "batch_size": batch_size,
        "num_iters": num_iters,
        "eval_interval": eval_interval,
        "num_eval_batches": num_eval_batches,
        "max_l2_norm": max_l2_norm,
        "max_learning_rate": max_learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cosine_cycle_iters,
        "weight_decay": args.weight_decay,
        **{key: value for key, value in experiment.items() if key != "run_name"},
    }
    save_config(config, run_dir)
    print(json.dumps(config, indent=2))

    train(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        batch_size=batch_size,
        context_length=context_length,
        num_iters=num_iters,
        eval_interval=eval_interval,
        eval_data=valid_data,
        num_eval_batches=num_eval_batches,
        max_l2_norm=max_l2_norm,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
        device=device,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
    )



if __name__ == "__main__":
    main()
