import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RUN_ROOT = Path("/data/leejt/cs336_assignment1/runs")
CONTROL_RUN = "tinystories_lr3e-4"

COLORS = {
    "baseline": "#111827",
    "blue": "#0072B2",
    "light_blue": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
}

GROUPS = [
    {
        "legend_title": "Architecture ablations",
        "runs": [
            ("Baseline", CONTROL_RUN, COLORS["baseline"], "-", 2.6),
            ("No RMSNorm", "tinystories_no_rmsnorm", COLORS["blue"], "--", 1.9),
            ("Post-Norm", "tinystories_post_norm", COLORS["orange"], "-.", 1.9),
            ("NoPE", "tinystories_nope", COLORS["green"], ":", 2.0),
            ("SiLU", "tinystories_silu", COLORS["purple"], (0, (5, 1)), 1.9),
            (
                "No RMSNorm, lr=1e-4",
                "tinystories_no_rmsnorm_lr1e-4",
                COLORS["red"],
                (0, (2, 1)),
                1.8,
            ),
        ],
    },
    {
        "legend_title": "Learning-rate sweep",
        "runs": [
            ("lr=1e-4", "tinystories_lr1e-4", COLORS["light_blue"], "--", 1.8),
            ("lr=2e-4", "tinystories_lr2e-4", COLORS["blue"], "-.", 1.8),
            ("lr=3e-4", CONTROL_RUN, COLORS["baseline"], "-", 2.6),
            ("lr=6e-4", "tinystories_lr6e-4", COLORS["green"], ":", 2.0),
            ("lr=1e-3", "tinystories_lr1e-3", COLORS["orange"], "--", 1.8),
            ("lr=3e-3", "tinystories_lr3e-3", COLORS["red"], "-.", 1.9),
        ],
    },
    {
        "legend_title": "Batch-size sweep",
        "runs": [
            ("batch=1", "tinystories_batch1", COLORS["purple"], "--", 1.8),
            ("batch=32", "tinystories_batch32", COLORS["blue"], "-.", 1.8),
            ("batch=64", "tinystories_batch64", COLORS["baseline"], "-", 2.6),
            ("batch=128", "tinystories_batch128", COLORS["orange"], "--", 1.9),
        ],
        "oom_label": "batch=192 (OOM)",
    },
]


def load_eval_metrics(path: Path) -> tuple[list[int], list[float]]:
    steps = []
    losses = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "eval_loss" in record:
                steps.append(record["iteration"])
                losses.append(record["eval_loss"])
    if not steps:
        raise ValueError(f"No evaluation metrics found in {path}")
    return steps, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TinyStories validation-loss curves.")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tinystories_ablations"))
    args = parser.parse_args()

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "text.usetex": False,
        }
    )

    loaded_groups = []
    all_losses = []
    for group in GROUPS:
        loaded_runs = []
        for label, run_name, color, linestyle, linewidth in group["runs"]:
            steps, losses = load_eval_metrics(args.run_root / run_name / "metrics.jsonl")
            loaded_runs.append((label, steps, losses, color, linestyle, linewidth))
            all_losses.extend(losses)
        loaded_groups.append((group, loaded_runs))

    y_min = min(all_losses)
    y_max = max(all_losses)
    y_margin = max(0.05, (y_max - y_min) * 0.04)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), sharex=True, sharey=True)
    x_ticks = [0, 5000, 10000, 15000, 20000]

    for panel_index, (group, loaded_runs) in enumerate(zip(GROUPS, loaded_groups)):
        ax = axes[panel_index]
        _, loaded_runs = loaded_runs
        for label, steps, losses, color, linestyle, linewidth in loaded_runs:
            ax.plot(
                steps,
                losses,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
                solid_capstyle="round",
            )
            ax.scatter([steps[-1]], [losses[-1]], color=color, s=22, zorder=4)

        ax.text(
            0.02,
            0.97,
            f"({chr(ord('a') + panel_index)})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
        )
        ax.set_xlim(0, 20000)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xticks(x_ticks)
        ax.set_xlabel("Training steps")
        ax.grid(True, axis="both")

        legend_handles, legend_labels = ax.get_legend_handles_labels()
        if "oom_label" in group:
            legend_handles.append(
                Line2D([], [], color="#6B7280", linestyle="--", linewidth=1.4)
            )
            legend_labels.append(group["oom_label"])
        ax.legend(
            legend_handles,
            legend_labels,
            title=group["legend_title"],
            title_fontsize=8.5,
            frameon=False,
            loc="upper right",
            handlelength=2.1,
            borderpad=0.2,
            labelspacing=0.35,
        )

    axes[0].set_ylabel("Validation loss")
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.16, top=0.98, wspace=0.12)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "ablation_loss_curves.pdf"
    png_path = args.output_dir / "ablation_loss_curves.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
