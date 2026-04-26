import csv
import json
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUTPUT_DIR = ROOT / "thesis_work" / "grafikonok"
DATA_DIR = OUTPUT_DIR / "data"

sys.path.insert(0, str(ROOT))
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/preference_distillation_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/preference_distillation_matplotlib_cache")

FULL_VOCAB_SIZE = 262_144

VOCABS = {
    "Sentiment": {"input": 68_328, "output": 525},
    "Math": {"input": 1_282, "output": 528},
    "Post generation": {"input": 72_133, "output": 26_659},
}

COLORS = {
    "blue": "#2563eb",
    "sky": "#38bdf8",
    "green": "#16a34a",
    "orange": "#f97316",
    "amber": "#f59e0b",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "pink": "#db2777",
    "gray": "#64748b",
    "light_gray": "#cbd5e1",
    "dark": "#111827",
    "muted": "#475569",
    "grid": "#e2e8f0",
    "paper": "#ffffff",
}


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT = {
    "axis": font(24),
    "label": font(22),
    "small": font(18),
    "tiny": font(15),
    "bold": font(22, True),
}


def color(name, alpha=255):
    rgb = ImageColor.getrgb(COLORS[name] if name in COLORS else name)
    return (*rgb, alpha)


def faded_on_paper(rgba, opacity=0.3):
    paper = ImageColor.getrgb(COLORS["paper"])
    return tuple(round(channel * opacity + paper[index] * (1 - opacity)) for index, channel in enumerate(rgba[:3])) + (255,)


def canvas(width=1600, height=900):
    image = Image.new("RGBA", (width, height), color("paper"))
    return image, ImageDraw.Draw(image)


def text_size(draw, text, text_font):
    box = draw.textbbox((0, 0), str(text), font=text_font)
    return box[2] - box[0], box[3] - box[1]


def text_center(draw, xy, text, text_font=FONT["small"], fill=None):
    x, y = xy
    width, height = text_size(draw, text, text_font)
    draw.text((x - width / 2, y - height / 2), text, font=text_font, fill=fill or color("dark"))


def save_image(image, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    image.save(path)
    print(f"Saved {path}")


def write_csv(filename, rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def training_entries(run_name):
    return read_jsonl(RUNS_DIR / run_name / "logs" / "training.jsonl")


def final_eval(run_name):
    evals = [entry for entry in training_entries(run_name) if entry.get("type") == "eval_epoch"]
    return evals[-1] if evals else None


def run_info(run_name):
    return read_json(RUNS_DIR / run_name / "info.json")


def matching_runs(pattern):
    return sorted(path.name for path in RUNS_DIR.glob(pattern) if (path / "info.json").exists())


def percent(value):
    return value * 100


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def std(values):
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = math.sqrt(((len(a) - 1) * std(a) ** 2 + (len(b) - 1) * std(b) ** 2) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (mean(a) - mean(b)) / pooled


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    denom = math.sqrt(std(a) ** 2 / len(a) + std(b) ** 2 / len(b))
    if denom == 0:
        return 0.0
    return (mean(a) - mean(b)) / denom


def draw_axes(draw, area, y_min, y_max, y_label="", x_label="", y_ticks=6):
    left, top, right, bottom = area
    for index in range(y_ticks):
        fraction = index / (y_ticks - 1)
        y = bottom - fraction * (bottom - top)
        value = y_min + fraction * (y_max - y_min)
        draw.line((left, y, right, y), fill=color("grid"), width=1)
        label = f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}"
        draw.text((left - 62, y - 10), label, font=FONT["tiny"], fill=color("muted"))
    draw.line((left, top, left, bottom), fill=color("dark"), width=2)
    draw.line((left, bottom, right, bottom), fill=color("dark"), width=2)
    if y_label:
        label_width, label_height = text_size(draw, y_label, FONT["small"])
        label_box = (left + 8, top + 8, left + label_width + 20, top + label_height + 16)
        draw.rectangle(label_box, fill=color("paper"))
        draw.text((left + 14, top + 10), y_label, font=FONT["small"], fill=color("muted"))
    if x_label:
        text_center(draw, ((left + right) / 2, bottom + 54), x_label, FONT["small"], color("muted"))


def draw_legend(draw, items, x, y):
    for label, item_color in items:
        draw.rectangle((x, y + 3, x + 24, y + 27), fill=item_color)
        draw.text((x + 34, y), label, font=FONT["small"], fill=color("dark"))
        y += 34


def draw_line_chart(draw, area, series, y_min, y_max, x_min=None, x_max=None, x_log=False, y_label="", x_label="", legend_x=None):
    left, top, right, bottom = area
    xs = [x for item in series for x, _ in item["points"]]
    if x_log:
        x_min = math.log10(min(xs)) if x_min is None else math.log10(x_min)
        x_max = math.log10(max(xs)) if x_max is None else math.log10(x_max)
    else:
        x_min = min(xs) if x_min is None else x_min
        x_max = max(xs) if x_max is None else x_max

    draw_axes(draw, area, y_min, y_max, y_label=y_label, x_label=x_label)

    if x_log:
        ticks = [tick for tick in [1, 5, 10, 20, 50, 100, 200, 500] if min(xs) <= tick <= max(xs)]
    else:
        ticks = [min(xs) + (max(xs) - min(xs)) * i / 4 for i in range(5)]
    for tick in ticks:
        transformed = math.log10(tick) if x_log else tick
        x = left + (transformed - x_min) / (x_max - x_min) * (right - left)
        draw.line((x, top, x, bottom), fill=color("grid"), width=1)
        text_center(draw, (x, bottom + 22), f"{tick:g}", FONT["tiny"], color("muted"))

    def map_point(point):
        x_value, y_value = point
        x_transformed = math.log10(x_value) if x_log else x_value
        x = left + (x_transformed - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
        return x, y

    legend_items = []
    for item in series:
        points = [map_point(point) for point in item["points"]]
        if len(points) > 1:
            draw.line(points, fill=item["color"], width=item.get("width", 4))
        for x, y in points:
            radius = item.get("radius", 6)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=item["color"], outline=color("paper"), width=2)
        legend_items.append((item["name"], item["color"]))
    draw_legend(draw, legend_items, legend_x or right + 45, top + 10)


def cleanup_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*.png"):
        path.unlink()
    for path in DATA_DIR.glob("*.csv"):
        path.unlink()


def plot_vocabulary_reduction():
    rows = [
        {"domain": domain, "full": FULL_VOCAB_SIZE, "input": values["input"], "output": values["output"]}
        for domain, values in VOCABS.items()
    ]
    image, draw = canvas()
    area = (130, 70, 1260, 760)
    draw_axes(draw, area, 0, 280_000, y_label="Tokens", x_label="Domain", y_ticks=8)
    labels = [row["domain"] for row in rows]
    series = [
        ("Gemma full", "full", color("gray")),
        ("Reduced input", "input", color("blue")),
        ("Reduced output", "output", color("orange")),
    ]
    group_width = (area[2] - area[0]) / len(rows)
    bar_width = 70
    for domain_index, row in enumerate(rows):
        center = area[0] + group_width * (domain_index + 0.5)
        for series_index, (_, key, fill) in enumerate(series):
            value = row[key]
            x0 = center - 130 + series_index * 86
            y0 = area[3] - value / 280_000 * (area[3] - area[1])
            draw.rectangle((x0, y0, x0 + bar_width, area[3]), fill=fill)
            label_y = y0 - 24 if value > 5_000 else area[3] - 54 - series_index * 22
            text_center(draw, (x0 + bar_width / 2, label_y), f"{value:,}", FONT["tiny"], color("dark"))
            if value <= 1_000:
                draw.line((x0 + bar_width / 2, area[3] - 5, x0 + bar_width / 2, label_y + 15), fill=fill, width=2)
        text_center(draw, (center, area[3] + 28), row["domain"], FONT["small"], color("muted"))
    draw_legend(draw, [(name, fill) for name, _, fill in series], 1320, 90)
    write_csv("01_vocabulary_reduction.csv", rows)
    save_image(image, "01_vocabulary_reduction.png")


def parameter_components(input_vocab, output_vocab, hidden=80, layers=5):
    return {
        "input_embedding": input_vocab * hidden,
        "attention": 4 * hidden * hidden * layers,
        "feed_forward": 12 * hidden * hidden * layers,
        "output_projection": output_vocab * hidden + output_vocab,
        "norm": 2 * hidden * layers + hidden,
    }


def plot_parameter_allocation():
    domains = [
        ("Sentiment", VOCABS["Sentiment"]["input"], VOCABS["Sentiment"]["output"]),
        ("Post generation", VOCABS["Post generation"]["input"], VOCABS["Post generation"]["output"]),
    ]
    scenarios = [
        ("No reduction", FULL_VOCAB_SIZE, FULL_VOCAB_SIZE),
        ("Input-only", None, FULL_VOCAB_SIZE),
        ("Input + output", None, None),
    ]
    components = [
        ("input_embedding", "Input embedding", color("blue")),
        ("attention", "Attention", color("green")),
        ("feed_forward", "Feed-forward", color("orange")),
        ("output_projection", "Output projection", color("purple")),
        ("norm", "Norm", color("gray")),
    ]
    rows = []
    image, draw = canvas()
    for domain_index, (domain, reduced_input, reduced_output) in enumerate(domains):
        y_base = 100 + domain_index * 390
        draw.text((90, y_base - 45), domain, font=FONT["bold"], fill=color("dark"))
        for scenario_index, (scenario, input_override, output_override) in enumerate(scenarios):
            input_vocab = reduced_input if input_override is None else input_override
            output_vocab = reduced_output if output_override is None else output_override
            values = parameter_components(input_vocab, output_vocab)
            total = sum(values.values())
            rows.append({"domain": domain, "scenario": scenario, "total_params": total, **values})
            x0, x1 = 310, 1260
            y = y_base + scenario_index * 95
            draw.text((90, y + 12), scenario, font=FONT["small"], fill=color("dark"))
            current = x0
            for key, _, fill in components:
                width = values[key] / total * (x1 - x0)
                if width > 0:
                    draw.rectangle((current, y, current + width, y + 42), fill=fill)
                if width > 62:
                    text_center(draw, (current + width / 2, y + 21), f"{values[key] / total * 100:.0f}%", FONT["tiny"], color("paper"))
                current += width
            draw.rectangle((x0, y, x1, y + 42), outline=color("dark"), width=1)
            draw.text((1285, y + 9), f"{total / 1_000_000:.2f}M", font=FONT["small"], fill=color("dark"))
    legend_x = 520
    for _, label, fill in components:
        draw.rectangle((legend_x, 26, legend_x + 20, 46), fill=fill)
        draw.text((legend_x + 28, 23), label, font=FONT["tiny"], fill=color("dark"))
        legend_x += text_size(draw, label, FONT["tiny"])[0] + 65
    write_csv("02_parameter_allocation.csv", rows)
    save_image(image, "02_parameter_allocation.png")


def plot_scaling_curve():
    names = [
        "tsc-scale-20h-2L-5M",
        "tsc-scale-40h-2L-10M",
        "tsc-scale-64h-2L-17M",
        "tsc-scale-128h-4L-36M",
        "tsc-scale-192h-6L-53M",
        "tsc-scale-256h-8L-74M",
        "tsc-scale-384h-12L-127M",
    ]
    rows = []
    for name in names:
        info = run_info(name)
        ev = final_eval(name)
        rows.append({
            "run": name,
            "parameters_m": info["model_info"]["total_parameters"] / 1_000_000,
            "teacher_forced_accuracy": percent(ev["teacher_forced_accuracy"]),
            "student_accuracy": percent(ev["student_accuracy"]),
            "task_accuracy": percent(ev["classification_accuracy"]),
        })
    image, draw = canvas()
    series = [
        {"name": "Teacher-forced token accuracy", "color": color("green"), "points": [(r["parameters_m"], r["teacher_forced_accuracy"]) for r in rows]},
        {"name": "Student-only token accuracy", "color": color("blue"), "points": [(r["parameters_m"], r["student_accuracy"]) for r in rows]},
        {"name": "Task accuracy", "color": color("orange"), "points": [(r["parameters_m"], r["task_accuracy"]) for r in rows]},
    ]
    draw_line_chart(draw, (120, 70, 1180, 760), series, 84, 100.5, x_log=True, y_label="Accuracy (%)", x_label="Parameters (millions, log scale)", legend_x=1230)
    write_csv("03_sentiment_scaling_curve.csv", rows)
    save_image(image, "03_sentiment_scaling_curve.png")


def plot_reduced_vs_full_vocab():
    pairs = [
        ("~5M", "tsc-scale-20h-2L-5M", "tsc-reduced-input-proj-5M-64h-8L"),
        ("~10M", "tsc-scale-40h-2L-10M", "tsc-reduced-input-proj-10M-128h-5L"),
        ("~36M", "tsc-scale-128h-4L-36M", "tsc-reduced-input-proj-36M-320h-9L"),
    ]
    rows = []
    for budget, full_run, reduced_run in pairs:
        full_ev = final_eval(full_run)
        reduced_ev = final_eval(reduced_run)
        rows.append({
            "budget": budget,
            "full_student": percent(full_ev["student_accuracy"]),
            "reduced_student": percent(reduced_ev["student_accuracy"]),
            "full_task": percent(full_ev["classification_accuracy"]),
            "reduced_task": percent(reduced_ev["classification_accuracy"]),
        })
    image, draw = canvas()
    panels = [
        ("Student-only token accuracy", "full_student", "reduced_student", (110, 80, 720, 750), 91, 94),
        ("Task accuracy", "full_task", "reduced_task", (840, 80, 1450, 750), 84, 89),
    ]
    for title, full_key, reduced_key, area, y_min, y_max in panels:
        draw.text((area[0], area[1] - 42), title, font=FONT["bold"], fill=color("dark"))
        draw_axes(draw, area, y_min, y_max, y_label="Accuracy (%)", x_label="Parameter budget")
        group_width = (area[2] - area[0]) / len(rows)
        for index, row in enumerate(rows):
            center = area[0] + group_width * (index + 0.5)
            for offset, key, fill in [(-32, full_key, color("gray")), (32, reduced_key, color("blue"))]:
                value = row[key]
                x0 = center + offset - 24
                y0 = area[3] - (value - y_min) / (y_max - y_min) * (area[3] - area[1])
                draw.rectangle((x0, y0, x0 + 48, area[3]), fill=fill)
                text_center(draw, (x0 + 24, y0 - 18), f"{value:.2f}", FONT["tiny"], color("dark"))
            text_center(draw, (center, area[3] + 26), row["budget"], FONT["small"], color("muted"))
    draw_legend(draw, [("Full input vocabulary", color("gray")), ("Reduced input vocabulary", color("blue"))], 575, 810)
    write_csv("04_reduced_vs_full_vocab.csv", rows)
    save_image(image, "04_reduced_vs_full_vocab.png")


LOSS_GROUPS = [
    ("CE", "purece", color("blue")),
    ("KL", "purekl", color("green")),
    ("KL/CE anneal", "kl99to50", color("orange")),
]


def loss_pattern(domain_prefix, group_key):
    if domain_prefix == "sentiment":
        return f"exp-tsc-8k-{group_key}-t1-48h-3L-3.4M-*"
    return f"exp-math-8k-{group_key}-t1-48h-3L-3.4M-*"


def strategy_evals(domain_prefix):
    output = {}
    for label, key, fill in LOSS_GROUPS:
        runs = matching_runs(loss_pattern(domain_prefix, key))
        output[label] = {"color": fill, "runs": runs, "evals": [final_eval(run) for run in runs]}
    return output


def plot_structured_loss_spread():
    image, draw = canvas()
    rows = []
    for panel_index, (domain_label, domain_prefix, y_min, y_max) in enumerate([
        ("Sentiment task accuracy", "sentiment", 58, 73),
        ("Math task accuracy", "math", 18, 26),
    ]):
        area = (110 + panel_index * 760, 100, 650 + panel_index * 760, 735)
        draw.text((area[0], area[1] - 45), domain_label, font=FONT["bold"], fill=color("dark"))
        draw_axes(draw, area, y_min, y_max, y_label="Task accuracy (%)", x_label="Loss strategy")
        groups = strategy_evals(domain_prefix)
        ce_values = [percent(ev["classification_accuracy"]) for ev in groups["CE"]["evals"]]
        for index, (label, _, fill) in enumerate(LOSS_GROUPS):
            values = [percent(ev["classification_accuracy"]) for ev in groups[label]["evals"]]
            avg = mean(values)
            deviation = std(values)
            x = area[0] + (area[2] - area[0]) * (index + 0.5) / 3
            y_avg = area[3] - (avg - y_min) / (y_max - y_min) * (area[3] - area[1])
            y_low = area[3] - (avg - deviation - y_min) / (y_max - y_min) * (area[3] - area[1])
            y_high = area[3] - (avg + deviation - y_min) / (y_max - y_min) * (area[3] - area[1])
            draw.line((x, y_low, x, y_high), fill=fill, width=4)
            draw.line((x - 14, y_low, x + 14, y_low), fill=fill, width=4)
            draw.line((x - 14, y_high, x + 14, y_high), fill=fill, width=4)
            draw.ellipse((x - 11, y_avg - 11, x + 11, y_avg + 11), fill=fill, outline=color("paper"), width=2)
            for seed_index, value in enumerate(values):
                jitter = (seed_index - (len(values) - 1) / 2) * 5
                y = area[3] - (value - y_min) / (y_max - y_min) * (area[3] - area[1])
                draw.ellipse((x + jitter - 4, y - 4, x + jitter + 4, y + 4), fill=color("dark", 90))
            text_center(draw, (x, area[3] + 26), label, FONT["small"], color("muted"))
            rows.append({
                "domain": domain_label.replace(" task accuracy", ""),
                "strategy": label,
                "n": len(values),
                "task_mean": avg,
                "task_std": deviation,
                "welch_t_vs_ce": welch_t(values, ce_values) if label != "CE" else 0,
                "cohen_d_vs_ce": cohen_d(values, ce_values) if label != "CE" else 0,
            })
    write_csv("05_structured_loss_spread.csv", rows)
    save_image(image, "05_structured_loss_spread.png")


def plot_accuracy_gap_seed_lines():
    rows = []
    metrics = [
        ("Teacher-forced", "teacher_forced_accuracy"),
        ("Student-only", "student_accuracy"),
        ("Task", "classification_accuracy"),
    ]
    filenames = {
        "CE": "06a_accuracy_gap_seed_lines_ce.png",
        "KL": "06b_accuracy_gap_seed_lines_kl.png",
        "KL/CE anneal": "06c_accuracy_gap_seed_lines_klce_annealing.png",
    }
    groups = strategy_evals("sentiment")
    sample_run = groups["CE"]["runs"][0]
    model_size = run_info(sample_run)["model_info"]["total_parameters"] / 1_000_000
    y_min, y_max = 58, 100

    for label, _, fill in LOSS_GROUPS:
        seed_fill = faded_on_paper(fill, 0.3)
        image, draw = canvas(width=1300, height=780)
        area = (145, 135, 1130, 625)
        draw.text((area[0], 52), f"Sentiment {model_size:.1f}M - {label}", font=FONT["bold"], fill=color("dark"))
        draw_axes(draw, area, y_min, y_max, y_label="Accuracy (%)", x_label="Metric")
        for metric_index, (metric_label, _) in enumerate(metrics):
            x = area[0] + (area[2] - area[0]) * (metric_index + 0.5) / len(metrics)
            text_center(draw, (x, area[3] + 28), metric_label, FONT["small"], color("muted"))

        run_values = []
        for run_name, ev in zip(groups[label]["runs"], groups[label]["evals"]):
            values = [percent(ev[key]) for _, key in metrics]
            run_values.append(values)
            points = []
            for metric_index, value in enumerate(values):
                x = area[0] + (area[2] - area[0]) * (metric_index + 0.5) / len(metrics)
                y = area[3] - (value - y_min) / (y_max - y_min) * (area[3] - area[1])
                points.append((x, y))
                rows.append({
                    "domain": "Sentiment",
                    "model_parameters_m": model_size,
                    "strategy": label,
                    "run": run_name,
                    "metric": metrics[metric_index][0],
                    "accuracy": value,
                    "summary": "seed",
                })
            draw.line(points, fill=seed_fill, width=2)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=seed_fill, outline=color("paper"), width=1)

        mean_values = [mean(values[index] for values in run_values) for index in range(len(metrics))]
        mean_points = []
        for metric_index, value in enumerate(mean_values):
            x = area[0] + (area[2] - area[0]) * (metric_index + 0.5) / len(metrics)
            y = area[3] - (value - y_min) / (y_max - y_min) * (area[3] - area[1])
            mean_points.append((x, y))
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=fill, outline=color("paper"), width=2)
            rows.append({
                "domain": "Sentiment",
                "model_parameters_m": model_size,
                "strategy": label,
                "run": "mean",
                "metric": metrics[metric_index][0],
                "accuracy": value,
                "summary": "mean",
            })
        draw.line(mean_points, fill=fill, width=5)
        draw.text((area[0], area[3] + 82), "faded lines: individual seeds", font=FONT["tiny"], fill=seed_fill)
        draw.text((area[0] + 290, area[3] + 82), "solid line: seed mean", font=FONT["tiny"], fill=fill)
        save_image(image, filenames[label])
    write_csv("06_accuracy_gap_seed_lines.csv", rows)


def smooth(values, window=25):
    if len(values) <= window:
        return values
    output = []
    running = sum(values[:window])
    for index in range(window, len(values) + 1):
        output.append(running / window)
        if index < len(values):
            running += values[index] - values[index - window]
    pad = [output[0]] * (window // 2)
    return pad + output


def mean_training_curve(pattern, metric, points=101):
    curves = []
    for run in matching_runs(pattern):
        batches = [entry for entry in training_entries(run) if entry.get("type") == "train_batch"]
        if not batches:
            continue
        values = smooth([percent(entry[metric]) if metric == "accuracy" else entry[metric] for entry in batches])
        sampled = []
        for point_index in range(points):
            source_index = min(len(values) - 1, round((len(values) - 1) * point_index / (points - 1)))
            sampled.append(values[source_index])
        curves.append(sampled)
    if not curves:
        return []
    return [(index, mean(curve[index] for curve in curves)) for index in range(points)]


def plot_loss_strategy_training_curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def moving_average(values, window):
        if len(values) < window:
            return values
        return [
            sum(values[index - window + 1:index + 1]) / window
            for index in range(window - 1, len(values))
        ]

    def plot_timeseries(axis, indices, data, light_color, dark_color, ylabel, title,
                        ma_window=40, ylim_data=None, default_padding=0.1,
                        is_percent=False, show_last_200_avg=True, ylim_padding_ratio=0.1):
        axis.plot(indices, data, color=light_color, alpha=0.5, linewidth=0.7)

        if len(data) >= ma_window:
            averaged = moving_average(data, ma_window)
            axis.plot(indices[ma_window - 1:], averaged, color=dark_color, linewidth=2, label=f"MA({ma_window})")

        if show_last_200_avg and len(data) >= 200:
            avg = sum(data[-200:]) / 200
            fmt = f"{avg:.1f}%" if is_percent else f"{avg:.4f}"
            axis.axhline(y=avg, color=dark_color, linewidth=1, alpha=0.25, linestyle="-", label=f"Last 200 avg: {fmt}")

        if ylim_data:
            min_value = min(ylim_data)
            max_value = max(ylim_data)
            padding = (max_value - min_value) * ylim_padding_ratio if max_value > min_value else default_padding
            low = max(0, min_value - padding)
            high = min(100, max_value + padding) if is_percent else max_value + padding
            axis.set_ylim([low, high])

        axis.set_xlabel("Batch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
        axis.grid(True, alpha=0.3)

    strategy_runs = [
        ("Pure CE", "short", "exp-tsc-8k-purece-t1-48h-3L-3.4M-1", "07a_pure_ce_training_curves.png"),
        ("Pure KL", "short", "exp-tsc-8k-purekl-t1-48h-3L-3.4M-1", "07b_pure_kl_training_curves.png"),
        ("KL/CE annealing", "short", "exp-tsc-8k-kl99to50-t1-48h-3L-3.4M-1", "07c_klce_annealing_training_curves.png"),
        ("Pure CE", "larger", "exp-purece-t1-1", "07d_large_pure_ce_training_curves.png"),
        ("Pure KL", "larger", "exp-purekl-t1-1", "07e_large_pure_kl_training_curves.png"),
        ("KL/CE annealing", "larger", "exp-kl99to50-t1-1", "07f_large_klce_annealing_training_curves.png"),
    ]
    rows = []
    for strategy_label, run_set, run_name, filename in strategy_runs:
        entries = training_entries(run_name)
        batches = [entry for entry in entries if entry.get("type") == "train_batch"]
        batch_indices = list(range(1, len(batches) + 1))
        losses = [batch["loss"] for batch in batches]
        kl_losses = [batch.get("kl_loss", 0) for batch in batches]
        ce_losses = [batch.get("ce_loss", 0) for batch in batches]
        accuracies = [batch["accuracy"] * 100 for batch in batches]
        ma_window = max(40, int(len(losses) * 0.03))
        last_90_start = int(len(losses) * 0.1)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        metrics = [
            ("Loss", losses, "lightcoral", "red", axes[0, 0], False),
            ("KL Loss", kl_losses, "plum", "purple", axes[0, 1], False),
            ("CE Loss", ce_losses, "lightskyblue", "blue", axes[1, 0], False),
            ("Accuracy", accuracies, "lightgreen", "green", axes[1, 1], True),
        ]
        for name, data, light_color, dark_color, axis, is_percent in metrics:
            data_90 = data[last_90_start:] if last_90_start < len(data) else data
            plot_timeseries(
                axis,
                batch_indices,
                data,
                light_color,
                dark_color,
                name,
                f"{name} - All ({len(data)} batches, y-axis: last 90%)",
                ma_window=ma_window,
                ylim_data=data_90,
                default_padding=5 if is_percent else 0.1,
                is_percent=is_percent,
                show_last_200_avg=True,
            )
            if name == "Accuracy" and len(data) >= 400:
                current = sum(data[-200:]) / 200
                previous = sum(data[-400:-200]) / 200
                axis.text(
                    0.02,
                    0.06,
                    f"Last-200 change: {current - previous:+.2f} pp",
                    transform=axis.transAxes,
                    fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
                )
            for batch_index, value in zip(batch_indices, data):
                rows.append({
                    "strategy": strategy_label,
                    "run_set": run_set,
                    "run": run_name,
                    "metric": name,
                    "batch": batch_index,
                    "value": value,
                })
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {OUTPUT_DIR / filename}")
    write_csv("07_loss_strategy_training_curves.csv", rows)


def draw_heatmap(draw, box, matrix, labels, title):
    left, top, right, bottom = box
    draw.text((left, top), title, font=FONT["bold"], fill=color("dark"))
    top += 34
    n = len(labels)
    cell = min((right - left - 92) / n, (bottom - top - 38) / n)
    grid_left = left + 92
    grid_top = top + 35
    max_value = max(max(row) for row in matrix) or 1
    for index, label in enumerate(labels):
        column_label = {
            "aggressive": "aggr.",
            "respectful": "respect.",
        }.get(label, label)
        draw.text((grid_left + index * cell + 2, top + 6), column_label[:10], font=FONT["tiny"], fill=color("muted"))
        draw.text((left + 2, grid_top + index * cell + cell / 2 - 9), label[:10], font=FONT["tiny"], fill=color("muted"))
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = value / max_value
            fill = (
                int(245 - 170 * intensity),
                int(248 - 115 * intensity),
                255,
                255,
            )
            x0 = grid_left + col_index * cell
            y0 = grid_top + row_index * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline=color("paper"))
            if value:
                text_center(draw, (x0 + cell / 2, y0 + cell / 2), str(int(value)), FONT["tiny"], color("dark"))


def plot_confusion_matrices():
    run_name = "tsc-scale-20h-2L-5M"
    ev = final_eval(run_name)
    confusion = ev["confusion_matrices"]
    image, draw = canvas(width=1240, height=800)
    boxes = {
        "tone": (45, 55, 585, 365),
        "sentiment": (660, 55, 1200, 365),
        "safety": (45, 405, 585, 715),
        "toxicity": (660, 405, 1200, 715),
    }
    rows = []
    for category, box in boxes.items():
        matrix_data = confusion[category]
        labels = list(matrix_data.keys())
        matrix = [[matrix_data[true].get(pred, 0) for pred in labels] for true in labels]
        draw_heatmap(draw, box, matrix, labels, category)
        for true_label, row in zip(labels, matrix):
            record = {"run": run_name, "category": category, "true_label": true_label}
            for pred_label, value in zip(labels, row):
                record[f"pred_{pred_label}"] = value
            rows.append(record)
    write_csv("08_reddit_confusion_matrices.csv", rows)
    save_image(image, "08_reddit_confusion_matrices.png")


def plot_postgen_topk():
    payload = read_json(ROOT / "top-k-accruacy-analsy" / "outputs" / "top_k_results.json")
    label_map = {
        "exp-postgen-3.8k-10ep-kl99to50-t5to3-48h-3L-14M": "KL/CE + temp",
        "exp-postgen-3.8k-10ep-purece-t1-48h-3L-14M": "Pure CE",
        "exp-postgen-3.8k-10ep-purekl-t5-48h-3L-14M": "Pure KL",
    }
    rows = []
    for result in payload["results"]:
        rows.append({
            "setting": label_map.get(result["run_name"], result["run_name"]),
            "top20_hit": percent(result["top_k_accuracy"]),
            "top20_overlap": percent(result["teacher_student_top_k_overlap"]),
            "mean_target_rank": result["mean_target_rank"],
            "examples": result["total_examples"],
            "steps": result["total_steps"],
        })
    image, draw = canvas(height=1000)
    panels = [
        ("Top-20 metrics", ("top20_hit", "top20_overlap"), (110, 80, 720, 740), 0, 80, "Percent (%)"),
        ("Mean target rank", ("mean_target_rank",), (880, 80, 1450, 740), 0, 1200, "Rank"),
    ]
    metric_colors = {
        "top20_hit": color("blue"),
        "top20_overlap": color("green"),
        "mean_target_rank": color("purple"),
    }
    metric_labels = {
        "top20_hit": "Top-20 target hit",
        "top20_overlap": "Top-20 overlap",
        "mean_target_rank": "Mean target rank",
    }
    for title, metrics, area, y_min, y_max, y_label in panels:
        draw.text((area[0], area[1] - 42), title, font=FONT["bold"], fill=color("dark"))
        draw_axes(draw, area, y_min, y_max, y_label=y_label, x_label="Strategy")
        group_width = (area[2] - area[0]) / len(rows)
        bar_width = 58 if len(metrics) == 2 else 72
        for row_index, row in enumerate(rows):
            center = area[0] + group_width * (row_index + 0.5)
            for metric_index, metric in enumerate(metrics):
                value = row[metric]
                x0 = center - (len(metrics) * bar_width) / 2 + metric_index * bar_width
                y0 = area[3] - (value - y_min) / (y_max - y_min) * (area[3] - area[1])
                draw.rectangle((x0, y0, x0 + bar_width * 0.82, area[3]), fill=metric_colors[metric])
                label = f"{value:.1f}" if value < 100 else f"{value:.0f}"
                text_center(draw, (x0 + bar_width * 0.41, y0 - 18), label, FONT["tiny"], color("dark"))
            text_center(draw, (center, area[3] + 28), row["setting"], FONT["small"], color("muted"))
    draw_legend(draw, [(metric_labels[key], metric_colors[key]) for key in ["top20_hit", "top20_overlap", "mean_target_rank"]], 520, 825)
    write_csv("09_postgen_topk_metrics.csv", rows)
    save_image(image, "09_postgen_topk_metrics.png")


def plot_domain_data_cost():
    rows = [
        {"domain": "Sentiment", "steps_per_example": 38.9, "output_vocab": 525, "logit_values": 20_400, "batch_mb": 5.9},
        {"domain": "Math", "steps_per_example": 32.0, "output_vocab": 528, "logit_values": 16_900, "batch_mb": 5.5},
        {"domain": "Post generation", "steps_per_example": 62.9, "output_vocab": 26_659, "logit_values": 1_680_000, "batch_mb": 512.0},
    ]
    image, draw = canvas()
    panels = [
        ("Logit values / example", "logit_values", (110, 80, 720, 740), 0, 1_800_000),
        ("Batch file size", "batch_mb", (880, 80, 1450, 740), 0, 560),
    ]
    for title, key, area, y_min, y_max in panels:
        draw.text((area[0], area[1] - 42), title, font=FONT["bold"], fill=color("dark"))
        draw_axes(draw, area, y_min, y_max, y_label=title, x_label="Domain")
        group_width = (area[2] - area[0]) / len(rows)
        for index, row in enumerate(rows):
            center = area[0] + group_width * (index + 0.5)
            value = row[key]
            y0 = area[3] - (value - y_min) / (y_max - y_min) * (area[3] - area[1])
            draw.rectangle((center - 38, y0, center + 38, area[3]), fill=color("orange" if key == "batch_mb" else "blue"))
            label = f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else (f"{value:.1f}" if value < 1000 else f"{value / 1000:.1f}k")
            text_center(draw, (center, y0 - 18), label, FONT["tiny"], color("dark"))
            text_center(draw, (center, area[3] + 28), row["domain"].replace(" ", "\n"), FONT["small"], color("muted"))
    write_csv("10_domain_data_cost.csv", rows)
    save_image(image, "10_domain_data_cost.png")


def write_graph_index():
    entries = [
        ("01_vocabulary_reduction.png", "Vocabulary reduction", "Theoretical reduced input/output vocabulary sizes by domain."),
        ("02_parameter_allocation.png", "Parameter allocation", "No reduction vs input-only vs input+output reduction for sentiment and post generation."),
        ("03_sentiment_scaling_curve.png", "Sentiment scaling", "Accuracy saturation across model sizes."),
        ("04_reduced_vs_full_vocab.png", "Reduced vs full input vocab", "Full and reduced-input student/task accuracy at comparable parameter budgets."),
        ("05_structured_loss_spread.png", "Structured loss spread", "Seed spread and standard deviation for task accuracy."),
        ("06a_accuracy_gap_seed_lines_ce.png", "CE accuracy gap", "Teacher-forced/student/task gap for sentiment CE seeds."),
        ("06b_accuracy_gap_seed_lines_kl.png", "KL accuracy gap", "Teacher-forced/student/task gap for sentiment KL seeds."),
        ("06c_accuracy_gap_seed_lines_klce_annealing.png", "KL/CE annealing accuracy gap", "Teacher-forced/student/task gap for sentiment annealing seeds."),
        ("07a_pure_ce_training_curves.png", "Pure CE training curves", "Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy."),
        ("07b_pure_kl_training_curves.png", "Pure KL training curves", "Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy."),
        ("07c_klce_annealing_training_curves.png", "KL/CE annealing training curves", "Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy."),
        ("07d_large_pure_ce_training_curves.png", "Larger pure CE training curves", "Same plotting style for the longer sentiment pure CE run."),
        ("07e_large_pure_kl_training_curves.png", "Larger pure KL training curves", "Same plotting style for the longer sentiment pure KL run."),
        ("07f_large_klce_annealing_training_curves.png", "Larger KL/CE annealing training curves", "Same plotting style for the longer sentiment KL/CE annealing run."),
        ("08_reddit_confusion_matrices.png", "Confusion matrices", "Sentiment category mistakes for the 5M-scale run."),
        ("09_postgen_topk_metrics.png", "Post generation top-k", "Free-form generation top-k tradeoff."),
        ("10_domain_data_cost.png", "Domain data cost", "Why post generation is much more expensive to compile."),
    ]
    lines = [
        "# Grafikonok indexe",
        "",
        "A script tobbnyire chart-only PNG-ket general. A 06a-c abrakban a modellmeret is latszik.",
        "",
    ]
    for index, (filename, title, note) in enumerate(entries, start=1):
        lines.append(f"{index}. `{filename}` - {title}: {note}")
    (OUTPUT_DIR / "GRAPH_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    cleanup_outputs()
    plot_vocabulary_reduction()
    plot_parameter_allocation()
    plot_scaling_curve()
    plot_reduced_vs_full_vocab()
    plot_structured_loss_spread()
    plot_accuracy_gap_seed_lines()
    plot_loss_strategy_training_curves()
    plot_confusion_matrices()
    plot_postgen_topk()
    plot_domain_data_cost()
    write_graph_index()


if __name__ == "__main__":
    main()
