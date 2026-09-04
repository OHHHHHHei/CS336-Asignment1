from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cs336_basics.nn import TransformerLM
from cs336_basics.tokenizer import load_tokenizer_files


def synchronize(device: torch.device) -> None:
    # CUDA kernel 默认异步执行，计时前后同步才能得到完整的 GPU 耗时。
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def run_generation(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    new_tokens: int,
    context_length: int,
    use_cache: bool,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    generated = prompt_ids.clone()
    step_times = []
    forward_times = []

    total_start = time.perf_counter()

    if use_cache:
        # 第一次前向处理整个 prompt，得到第一枚 token 的 logits 和各层 cache。
        step_start = time.perf_counter()
        forward_start = time.perf_counter()
        logits, past_key_values = model(
            generated[:, -context_length:],
            use_cache=True,
        )
        synchronize(device)
        prefill_ms = (time.perf_counter() - forward_start) * 1000
        forward_times.append(prefill_ms)

        next_token_id = greedy_next_token(logits)
        synchronize(device)
        step_times.append((time.perf_counter() - step_start) * 1000)
        generated = torch.cat((generated, next_token_id), dim=1)

        for _ in range(1, new_tokens):
            step_start = time.perf_counter()
            cache_length = past_key_values[0].valid_length
            if cache_length < context_length:
                logits, past_key_values = model(
                    next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # 窗口满后重新建立最近上下文的 cache，并把 RoPE 位置重新从 0 编号。
                logits, past_key_values = model(
                    generated[:, -context_length:],
                    use_cache=True,
                )
            synchronize(device)
            forward_times.append((time.perf_counter() - step_start) * 1000)

            next_token_id = greedy_next_token(logits)
            synchronize(device)
            step_times.append((time.perf_counter() - step_start) * 1000)
            generated = torch.cat((generated, next_token_id), dim=1)
    else:
        # 对照路径每一步都重新计算最近的上下文，模拟没有 KV cache 的生成。
        for step in range(new_tokens):
            step_start = time.perf_counter()
            forward_start = time.perf_counter()
            logits = model(generated[:, -context_length:])
            synchronize(device)
            forward_ms = (time.perf_counter() - forward_start) * 1000
            forward_times.append(forward_ms)
            if step == 0:
                prefill_ms = forward_ms

            next_token_id = greedy_next_token(logits)
            synchronize(device)
            step_times.append((time.perf_counter() - step_start) * 1000)
            generated = torch.cat((generated, next_token_id), dim=1)

    synchronize(device)
    total_ms = (time.perf_counter() - total_start) * 1000
    decode_times = step_times[1:]
    decode_total_ms = sum(decode_times)

    metrics = {
        "generated_tokens": float(new_tokens),
        "total_ms": total_ms,
        "prefill_ms": prefill_ms,
        "first_token_ms": step_times[0],
        "avg_token_ms": mean(step_times),
        "avg_decode_token_ms": mean(decode_times) if decode_times else step_times[0],
        "decode_total_ms": decode_total_ms,
        "tokens_per_second": new_tokens / total_ms * 1000,
        "decode_tokens_per_second": (
            (new_tokens - 1) / decode_total_ms * 1000 if decode_times else 0.0
        ),
        "avg_forward_ms": mean(forward_times),
    }
    return metrics, generated.cpu()


def measure_method(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    new_tokens: int,
    context_length: int,
    use_cache: bool,
    device: torch.device,
    warmup_runs: int,
    runs: int,
) -> tuple[dict[str, float], torch.Tensor]:
    for _ in range(warmup_runs):
        with torch.inference_mode():
            run_generation(model, prompt_ids, new_tokens, context_length, use_cache, device)

    measured = []
    output = prompt_ids.cpu()
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            metrics, output = run_generation(
                model,
                prompt_ids,
                new_tokens,
                context_length,
                use_cache,
                device,
            )

        if device.type == "cuda":
            synchronize(device)
            metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
            metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2
        else:
            metrics["peak_allocated_mb"] = 0.0
            metrics["peak_reserved_mb"] = 0.0
        measured.append(metrics)

    metric_names = measured[0].keys()
    averaged = {name: mean(item[name] for item in measured) for name in metric_names}
    averaged["peak_allocated_mb"] = max(item["peak_allocated_mb"] for item in measured)
    averaged["peak_reserved_mb"] = max(item["peak_reserved_mb"] for item in measured)
    return averaged, output


def build_model(
    device: torch.device,
    checkpoint_path: Path,
    context_length: int,
) -> tuple[TransformerLM, dict]:
    model = TransformerLM(
        vocab_size=10000,
        d_model=512,
        num_layers=4,
        n_heads=16,
        d_ff=1344,
        context_length=context_length,
        theta=10000.0,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def plot_results(results: list[dict], output_dir: Path) -> None:
    colors = {"kv_cache": "#0072B2", "no_kv_cache": "#D55E00"}
    labels = {"kv_cache": "KV cache", "no_kv_cache": "No KV cache"}
    x = [item["new_tokens"] for item in results]

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    panels = [
        ("total_ms", "Total latency (ms)"),
        ("avg_decode_token_ms", "Average decode token latency (ms)"),
        ("decode_tokens_per_second", "Decode throughput (tokens/s)"),
        ("peak_allocated_mb", "Peak allocated memory (MB)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, (metric_name, ylabel) in zip(axes.flat, panels):
        for method in ("kv_cache", "no_kv_cache"):
            y = [item[method][metric_name] for item in results]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors[method],
                label=labels[method],
            )
        ax.set_xlabel("Generated tokens")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(True, axis="both")

    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("KV Cache Inference Benchmark", y=0.995)
    fig.tight_layout()

    for extension in ("png", "pdf"):
        output_path = output_dir / f"kv_cache_benchmark.{extension}"
        fig.savefig(output_path, format=extension)
        print(f"已保存图片：{output_path}")
    plt.close(fig)


def write_report(
    report_path: Path,
    config: dict,
    results: list[dict],
) -> None:
    lines = [
        "# KV Cache 推理性能对比",
        "",
        "本次测试使用同一个 TinyStories checkpoint、同一个 prompt 和贪心解码。",
        "`KV cache` 使用预分配的静态 K/V 空间；`No KV cache` 每一步重新计算最近上下文。",
        "",
        "## 测试设置",
        "",
        f"- checkpoint：`{config['checkpoint']}`",
        f"- cache 实现：`{config['cache_type']}`",
        f"- prompt：`{config['prompt']}`",
        f"- 上下文长度：`{config['context_length']}`",
        f"- 生成长度：`{config['lengths']}`",
        f"- 设备：`{config['device']}`",
        f"- warmup：`{config['warmup_runs']}` 次，正式测量：`{config['runs']}` 次",
        "",
        "## 结果",
        "",
        "| 生成 token 数 | KV cache 总耗时 (ms) | 无 cache 总耗时 (ms) | KV cache 平均 decode (ms/token) | 无 cache 平均 decode (ms/token) | KV cache 吞吐 (token/s) | 无 cache 吞吐 (token/s) | 输出一致 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for item in results:
        cached = item["kv_cache"]
        full = item["no_kv_cache"]
        lines.append(
            f"| {item['new_tokens']} | {cached['total_ms']:.2f} | {full['total_ms']:.2f} | "
            f"{cached['avg_decode_token_ms']:.2f} | {full['avg_decode_token_ms']:.2f} | "
            f"{cached['decode_tokens_per_second']:.2f} | {full['decode_tokens_per_second']:.2f} | "
            f"{str(item['outputs_match'])} |"
        )

    lines.extend(
        [
            "",
            "`avg decode token latency` 从第二个生成 token 开始统计，排除了 prompt prefill。",
            "峰值显存包含模型参数、临时张量和 KV cache 本身。",
            "",
            "## 性能测试文件",
            "",
            "| 文件 | 类型 | 说明 |",
            "| --- | --- | --- |",
        "| `cs336_basics/generate.py` | 修改 | 增加 KV cache / 全量重算开关 |",
        "| `cs336_basics/nn.py` | 修改 | 使用每层预分配的静态 K/V 空间 |",
            "| `scripts/benchmark_kv_cache.py` | 新建 | 可复现的延迟、吞吐和显存测量脚本 |",
            "| `outputs/tinystories_base/kv_cache_benchmark.json` | 生成 | 原始测量结果 |",
            "| `outputs/tinystories_base/kv_cache_benchmark.png` | 生成 | 四项指标对比图 |",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已保存中文报告：{report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TinyStories KV cache inference.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("/data/leejt/cs336_assignment1/runs/tinystories_base/checkpoint.pt"),
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("/data/leejt/cs336_assignment1/data/TinyStoriesV2-GPT4-train"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tinystories_base"))
    parser.add_argument("--report-path", type=Path, default=Path("profile_output/kv_cache_benchmark.md"))
    parser.add_argument("--prompt", default="Hello, she said")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--lengths", default="16,32,64,128,192")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    lengths = [int(value) for value in args.lengths.split(",")]

    tokenizer = load_tokenizer_files(
        vocab_path=args.tokenizer_dir / "vocab.json",
        merges_path=args.tokenizer_dir / "merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    model, checkpoint = build_model(device, args.checkpoint_path, args.context_length)

    config = {
        "checkpoint": str(args.checkpoint_path),
        "checkpoint_iteration": checkpoint["iteration"],
        "prompt": args.prompt,
        "prompt_tokens": int(prompt_ids.shape[-1]),
        "context_length": args.context_length,
        "lengths": lengths,
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "device": str(device),
        "cache_type": "preallocated_static_kv_cache",
    }
    print(f"已加载 checkpoint：第 {checkpoint['iteration']} 次迭代")
    print(f"开始测试设备：{device}，prompt token 数：{prompt_ids.shape[-1]}")

    results = []
    for length in lengths:
        print(f"\n生成长度：{length} token")
        cached_metrics, cached_output = measure_method(
            model,
            prompt_ids,
            length,
            args.context_length,
            True,
            device,
            args.warmup_runs,
            args.runs,
        )
        full_metrics, full_output = measure_method(
            model,
            prompt_ids,
            length,
            args.context_length,
            False,
            device,
            args.warmup_runs,
            args.runs,
        )
        outputs_match = torch.equal(cached_output, full_output)
        result = {
            "new_tokens": length,
            "outputs_match": outputs_match,
            "speedup": full_metrics["total_ms"] / cached_metrics["total_ms"],
            "kv_cache": cached_metrics,
            "no_kv_cache": full_metrics,
        }
        results.append(result)
        print(
            f"  KV cache：{cached_metrics['total_ms']:.2f} ms，"
            f"{cached_metrics['decode_tokens_per_second']:.2f} token/s，"
            f"峰值显存 {cached_metrics['peak_allocated_mb']:.2f} MB"
        )
        print(
            f"  无 cache：{full_metrics['total_ms']:.2f} ms，"
            f"{full_metrics['decode_tokens_per_second']:.2f} token/s，"
            f"峰值显存 {full_metrics['peak_allocated_mb']:.2f} MB"
        )
        print(f"  输出一致：{outputs_match}，总耗时加速比：{result['speedup']:.2f}x")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "kv_cache_benchmark.json"
    result_path.write_text(
        json.dumps({"config": config, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n已保存原始结果：{result_path}")
    plot_results(results, args.output_dir)
    write_report(args.report_path, config, results)


if __name__ == "__main__":
    main()
