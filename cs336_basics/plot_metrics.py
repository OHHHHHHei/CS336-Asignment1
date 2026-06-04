import json
from pathlib import Path


def load_metrics(path: Path) -> list[dict]:
    metrics = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def scale_points(xs: list[float], ys: list[float], width: int, height: int, margin: int) -> list[tuple[float, float]]:
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    points = []
    for x, y in zip(xs, ys):
        px = margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
        py = height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)
        points.append((px, py))
    return points


def polyline(points: list[tuple[float, float]], color: str, width: float = 1.5, opacity: float = 1.0) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" opacity="{opacity}" />'
    )


def circles(points: list[tuple[float, float]], color: str, radius: float = 3.0) -> str:
    return "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" />'
        for x, y in points
    )


def render_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict],
    width: int = 1000,
    height: int = 520,
    margin: int = 64,
) -> None:
    all_xs = [x for item in series for x in item["xs"]]
    all_ys = [y for item in series for y in item["ys"]]
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="22">{title}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222" />',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>',
        f'<text x="18" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 18 {height / 2})">{y_label}</text>',
        f'<text x="{margin}" y="{height - margin + 24}" text-anchor="middle" font-family="Arial" font-size="12">{x_min:.0f}</text>',
        f'<text x="{width - margin}" y="{height - margin + 24}" text-anchor="middle" font-family="Arial" font-size="12">{x_max:.0f}</text>',
        f'<text x="{margin - 10}" y="{height - margin + 4}" text-anchor="end" font-family="Arial" font-size="12">{y_min:.4g}</text>',
        f'<text x="{margin - 10}" y="{margin + 4}" text-anchor="end" font-family="Arial" font-size="12">{y_max:.4g}</text>',
    ]

    legend_x = width - margin - 170
    legend_y = margin
    for idx, item in enumerate(series):
        points = scale_points(item["xs"], item["ys"], width, height, margin)
        elements.append(polyline(points, item["color"], width=item.get("width", 1.5), opacity=item.get("opacity", 1.0)))
        if item.get("markers"):
            elements.append(circles(points, item["color"], radius=item.get("radius", 3.0)))
        y = legend_y + idx * 24
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{item["color"]}" stroke-width="3" />')
        elements.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial" font-size="13">{item["label"]}</text>')

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    run_dir = Path("/data/leejt/cs336_assignment1/runs/tinystories_base")
    output_dir = Path("outputs/tinystories_base")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(run_dir / "metrics.jsonl")

    iterations = [m["iteration"] for m in metrics]
    train_losses = [m["train_loss"] for m in metrics]
    learning_rates = [m["learning_rate"] for m in metrics]

    eval_iterations = [m["iteration"] for m in metrics if "eval_loss" in m]
    eval_losses = [m["eval_loss"] for m in metrics if "eval_loss" in m]

    render_svg(
        output_dir / "loss_curve.svg",
        "TinyStories Training Loss",
        "iteration",
        "loss",
        [
            {"xs": iterations, "ys": train_losses, "label": "train loss", "color": "#2563eb", "opacity": 0.65},
            {"xs": eval_iterations, "ys": eval_losses, "label": "eval loss", "color": "#dc2626", "width": 2.0, "markers": True},
        ],
    )

    render_svg(
        output_dir / "lr_curve.svg",
        "Learning Rate Schedule",
        "iteration",
        "learning rate",
        [
            {"xs": iterations, "ys": learning_rates, "label": "learning rate", "color": "#16a34a", "width": 2.0},
        ],
    )

    print(f"Saved {output_dir / 'loss_curve.svg'}")
    print(f"Saved {output_dir / 'lr_curve.svg'}")


if __name__ == "__main__":
    main()
