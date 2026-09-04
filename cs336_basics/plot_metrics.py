from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def load_metrics(path: Path) -> list[dict]:
    metrics = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def configure_plot_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "svg.fonttype": "none",
        }
    )


def render_training_curves(metrics: list[dict], output_dir: Path) -> None:
    configure_plot_style()

    steps = np.asarray([item["iteration"] for item in metrics])
    train_losses = np.asarray([item["train_loss"] for item in metrics])
    learning_rates = np.asarray([item["learning_rate"] for item in metrics])
    eval_steps = np.asarray([item["iteration"] for item in metrics if "eval_loss" in item])
    eval_losses = np.asarray([item["eval_loss"] for item in metrics if "eval_loss" in item])
    train_color = "#0072B2"
    eval_color = "#D55E00"
    learning_rate_color = "#009E73"
    grid_color = "#D1D5DB"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.2),
        sharex=True,
    )
    loss_ax, lr_ax = axes

    loss_ax.plot(
        steps,
        train_losses,
        color=train_color,
        linewidth=0.75,
        alpha=0.28,
        label="Train loss (raw)",
    )
    loss_ax.plot(
        eval_steps,
        eval_losses,
        color=eval_color,
        linewidth=2.0,
        marker="o",
        markersize=3.2,
        markeredgewidth=0,
        label="Validation loss",
        zorder=3,
    )
    loss_ax.scatter([eval_steps[-1]], [eval_losses[-1]], color=eval_color, s=26, zorder=4)
    loss_ax.annotate(
        f"{eval_losses[-1]:.3f}",
        xy=(eval_steps[-1], eval_losses[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        color=eval_color,
        fontsize=9,
        ha="right",
    )
    loss_ax.set_ylabel("Loss")
    loss_ax.legend(frameon=False, loc="upper right", handlelength=2.4)

    lr_ax.plot(
        steps,
        learning_rates,
        color=learning_rate_color,
        linewidth=2.1,
        label="Learning rate",
    )
    lr_ax.scatter([steps[-1]], [learning_rates[-1]], color=learning_rate_color, s=26, zorder=4)
    lr_ax.annotate(
        f"{learning_rates[-1]:.1e}",
        xy=(steps[-1], learning_rates[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        color=learning_rate_color,
        fontsize=9,
        ha="right",
    )
    lr_ax.set_ylabel(r"Learning rate ($\times 10^{-4}$)")
    lr_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e4:.1f}"))

    max_step = int(steps[-1])
    x_ticks = np.linspace(0, max_step, 5, dtype=int)
    for ax in axes:
        ax.set_xlim(0, max_step)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: "0" if value == 0 else f"{value / 1000:.0f}k")
        )
        ax.grid(True, axis="y", color=grid_color)
        ax.grid(False, axis="x")
        ax.tick_params(axis="both", length=3)
        ax.set_xlabel("Training step")

    loss_ax.text(0.01, 0.98, "(a)", transform=loss_ax.transAxes, va="top", fontweight="bold")
    lr_ax.text(0.01, 0.98, "(b)", transform=lr_ax.transAxes, va="top", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.96, wspace=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        output_path = output_dir / f"training_curves.{extension}"
        fig.savefig(output_path, format=extension, facecolor="white")
        print(f"已保存图片：{output_path}")
    plt.close(fig)


def main() -> None:
    run_dir = Path("/data/leejt/cs336_assignment1/runs/tinystories_base")
    output_dir = Path("outputs/tinystories_base")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(run_dir / "metrics.jsonl")

    render_training_curves(metrics, output_dir)


if __name__ == "__main__":
    main()
