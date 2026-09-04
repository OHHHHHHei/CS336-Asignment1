from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS = [
    ("Baseline", "tinystories_base", "#111827", "-", 2.6),
    ("No RMSNorm", "tinystories_no_rmsnorm", "#0072B2", "--", 2.0),
    ("Post-Norm", "tinystories_post_norm", "#D55E00", "-.", 2.0),
    ("NoPE", "tinystories_nope", "#009E73", ":", 2.0),
    ("SiLU", "tinystories_silu", "#CC79A7", (0, (5, 1)), 2.0),
]


def load_eval_metrics(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "eval_loss" in record:
                steps.append(int(record["iteration"]))
                losses.append(float(record["eval_loss"]))
    if not steps:
        raise ValueError(f"No evaluation records found in {path}")
    return steps, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TinyStories ablation validation curves.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/data/leejt/cs336_assignment1/runs"),
        help="Directory containing tinystories_<experiment> run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tinystories_ablations"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for label, run_name, color, linestyle, linewidth in EXPERIMENTS:
        steps, losses = load_eval_metrics(args.run_root / run_name / "metrics.jsonl")
        final_loss = losses[-1]
        ax.plot(
            steps,
            losses,
            label=f"{label} ({final_loss:.3f})",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        ax.scatter([steps[-1]], [final_loss], color=color, s=22, zorder=3)

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(0, 20_000)
    ax.set_xticks([0, 5_000, 10_000, 15_000, 20_000])
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    for extension in ("pdf", "png"):
        output_path = args.output_dir / f"ablation_validation_loss.{extension}"
        fig.savefig(output_path, format=extension)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
