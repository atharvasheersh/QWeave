"""Render benchmark-v2 learning and method figures with Pillow."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont


COLORS = {"gnn_ppo": "#2463EB", "mlp_ppo": "#E27618"}


def _font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _quantile(values, q: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return float(values[low] * (1 - fraction) + values[high] * fraction)


def _canvas(title: str, subtitle: str):
    image = Image.new("RGB", (1800, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 45), title, font=_font(38, True), fill="#111827")
    draw.text((90, 98), subtitle, font=_font(22), fill="#4B5563")
    return image, draw


def _axes(draw, box, x_ticks, y_ticks, x_label, y_label):
    left, top, right, bottom = box
    draw.line((left, bottom, right, bottom), fill="#111827", width=3)
    draw.line((left, top, left, bottom), fill="#111827", width=3)
    for y, label in y_ticks:
        draw.line((left, y, right, y), fill="#E5E7EB", width=2)
        draw.text((left - 18, y), label, anchor="rm", font=_font(18), fill="#374151")
    for x, label in x_ticks:
        draw.line((x, bottom, x, bottom + 8), fill="#111827", width=2)
        draw.text((x, bottom + 18), label, anchor="ma", font=_font(18), fill="#374151")
    draw.text(((left + right) / 2, bottom + 75), x_label, anchor="ma",
              font=_font(21, True), fill="#111827")
    draw.text((left, top - 28), y_label, anchor="ls",
              font=_font(21, True), fill="#111827")


def learning_curves(payload: dict, output: Path) -> None:
    image, draw = _canvas(
        "QWeave benchmark v2 learning curves",
        "Median recent episode return across five seeds; shaded band is the interquartile range")
    box = (150, 175, 1710, 900)
    grouped = {}
    for item in payload["training"]:
        for point in item["training_curve"]:
            grouped.setdefault((item["method"], point["steps"]), []).append(
                point["mean_recent_return"])
    steps = sorted({step for _, step in grouped})
    series = {}
    for method in COLORS:
        series[method] = [
            (step, _quantile(grouped[(method, step)], 0.25),
             statistics.median(grouped[(method, step)]),
             _quantile(grouped[(method, step)], 0.75)) for step in steps]
    values = [value for rows in series.values() for _, low, mid, high in rows
              for value in (low, mid, high)]
    y_min = min(values) - 5
    y_max = max(values) + 5
    sx = lambda value: box[0] + (value - steps[0]) / (steps[-1] - steps[0]) * (box[2] - box[0])
    sy = lambda value: box[3] - (value - y_min) / (y_max - y_min) * (box[3] - box[1])
    x_values = list(range(0, 4097, 512))
    y_values = [y_min + index * (y_max - y_min) / 5 for index in range(6)]
    _axes(draw, box, [(sx(max(steps[0], x)), str(x)) for x in x_values if x >= steps[0]],
          [(sy(y), f"{y:.0f}") for y in y_values], "Environment steps", "Recent return")
    for method, rows in series.items():
        upper = [(sx(step), sy(high)) for step, _, _, high in rows]
        lower = [(sx(step), sy(low)) for step, low, _, _ in reversed(rows)]
        shade = "#BED0FA" if method == "gnn_ppo" else "#F8D5B1"
        draw.polygon(upper + lower, fill=shade)
        points = [(sx(step), sy(mid)) for step, _, mid, _ in rows]
        draw.line(points, fill=COLORS[method], width=6, joint="curve")
    for index, (method, color) in enumerate(COLORS.items()):
        x, y = 1220 + index * 240, 135
        draw.line((x, y, x + 50, y), fill=color, width=7)
        draw.text((x + 62, y), "GNN-PPO" if method == "gnn_ppo" else "No-message PPO",
                  anchor="lm", font=_font(19, True), fill="#111827")
    image.save(output, dpi=(180, 180))


def method_medians(payload: dict, output: Path) -> None:
    image, draw = _canvas(
        "Held-out routing results",
        "Case-level medians after collapsing seeds; lower depth and SWAP count are better")
    methods = ["basic", "weighted", "sabre", "gnn_ppo", "mlp_ppo", "gnn_untrained"]
    labels = ["Basic", "Weighted", "SABRE", "GNN-PPO", "No-message", "Untrained"]
    rows = {row["method"]: row for row in payload["analysis"]["method_summary"]}
    panels = [(100, 190, 850, 890, "Median depth", "depth"),
              (950, 190, 1700, 890, "Median SWAP count", "swap_count")]
    palette = ["#6B7280", "#059669", "#7C3AED", "#2463EB", "#E27618", "#B91C1C"]
    for left, top, right, bottom, title, metric in panels:
        values = [rows[method][metric]["median"] for method in methods]
        maximum = max(values) * 1.12
        draw.text(((left + right) / 2, top - 55), title, anchor="ma",
                  font=_font(25, True), fill="#111827")
        draw.line((left, bottom, right, bottom), fill="#111827", width=3)
        draw.line((left, top, left, bottom), fill="#111827", width=3)
        for index in range(5):
            value = maximum * index / 4
            y = bottom - (bottom - top) * index / 4
            draw.line((left, y, right, y), fill="#E5E7EB", width=2)
            draw.text((left - 15, y), f"{value:.0f}", anchor="rm",
                      font=_font(17), fill="#374151")
        width = (right - left) / len(methods)
        for index, (label, value, color) in enumerate(zip(labels, values, palette)):
            x0 = left + index * width + 18
            x1 = left + (index + 1) * width - 18
            y = bottom - value / maximum * (bottom - top)
            draw.rectangle((x0, y, x1, bottom), fill=color)
            draw.text(((x0 + x1) / 2, y - 12), f"{value:.1f}", anchor="ms",
                      font=_font(18, True), fill="#111827")
            draw.text(((x0 + x1) / 2, bottom + 24), label, anchor="ma",
                      font=_font(16), fill="#374151")
    image.save(output, dpi=(180, 180))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--destination", type=Path,
                        default=Path("results/benchmark_v2"))
    args = parser.parse_args()
    payload = json.loads(args.raw.read_text(encoding="utf-8"))
    figures = args.destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    learning_curves(payload, figures / "learning_curves.png")
    method_medians(payload, figures / "method_medians.png")
    sums = []
    for file in sorted(path for path in args.destination.rglob("*") if path.is_file()):
        if file.name == "SHA256SUMS.txt":
            continue
        sums.append(f"{sha256(file.read_bytes()).hexdigest()}  "
                    f"{file.relative_to(args.destination).as_posix()}")
    (args.destination / "SHA256SUMS.txt").write_text(
        "\n".join(sums) + "\n", encoding="utf-8")
    print(figures)


if __name__ == "__main__":
    main()
