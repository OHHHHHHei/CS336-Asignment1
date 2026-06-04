from pathlib import Path
import numpy as np
import torch

from cs336_basics.data import load_tokenized_data
from cs336_basics.nn import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.train import train

def main() -> None:
    # 数据加载部分
    data_dir = Path("/data/leejt/cs336_assignment1/data/TinyStoriesV2-GPT4-tokenized")
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

    batch_size = 64
    num_iters = 20000
    eval_interval = 500
    num_eval_batches = 50

    max_l2_norm = 1.0
    max_learning_rate = 3e-4
    min_learning_rate = 3e-5
    warmup_iters = 1000
    cosine_cycle_iters = num_iters

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
    )

    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=0.01,
    )

    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    run_dir = "/data/leejt/cs336_assignment1/runs/tinystories_base"
    # 确保运行目录存在
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    # 检查点路径
    checkpoint_path = f"{run_dir}/checkpoint.pt"

    train(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        batch_size=batch_size,
        context_length=context_length,
        num_iters=num_iters,
        eval_interval=eval_interval,
        eval_data = valid_data,
        num_eval_batches=num_eval_batches,
        max_l2_norm=max_l2_norm,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
        device=device,
        checkpoint_path=checkpoint_path,
        run_dir = run_dir,
    )



if __name__ == "__main__":
    main()
