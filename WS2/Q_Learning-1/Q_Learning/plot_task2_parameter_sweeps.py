"""Create publication figures for the Task 2 alpha and lambda sweeps."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plot_tensorboard_results import LOG_ROOT, RESULTS_DIR, save_figure, style_axis


SEEDS = range(5)
T_CRITICAL_95_DF4 = 2.776
WINDOW = 50
FINAL_EPISODES = 100
EARLY_EPISODES = 300
PERFORMANCE_THRESHOLD = -20.0
TAGS = ("Episode_Return", "Episode_Length", "Cliff_Falls")

SWEEPS = {
    "q2": {
        "parameter": "alpha",
        "values": (0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 1.00),
        "symbol": r"$\alpha$",
    },
    "q3": {
        "parameter": "lambda",
        "values": (0.00, 0.10, 0.30, 0.50, 0.70, 0.90, 1.00),
        "symbol": r"$\lambda$",
    },
}


def value_tag(value):
    return f"{value:.2f}".replace(".", "p")


def display_value(value):
    return f"{value:.2f}"


def select_runs(task):
    parameter = SWEEPS[task]["parameter"]
    pattern = re.compile(
        rf"task2_{task}_{parameter}(?P<value>\d+p\d+)_seed(?P<seed>\d+)"
    )
    selected = {}
    for directory in LOG_ROOT.iterdir():
        if not directory.is_dir() or not (directory / "q_table.npz").is_file():
            continue
        match = pattern.search(directory.name)
        if match is None:
            continue
        key = (match.group("value"), int(match.group("seed")))
        if key not in selected or directory.stat().st_mtime > selected[key].stat().st_mtime:
            selected[key] = directory

    missing = []
    for value in SWEEPS[task]["values"]:
        for seed in SEEDS:
            key = (value_tag(value), seed)
            if key not in selected:
                missing.append(f"{parameter}={value:.2f}, seed={seed}")
    if missing:
        raise RuntimeError("Missing completed sweep runs: " + ", ".join(missing))
    return selected


def load_sweep(task):
    selected = select_runs(task)
    data = {}
    for value in SWEEPS[task]["values"]:
        tag = value_tag(value)
        data[value] = {}
        for seed in SEEDS:
            accumulator = EventAccumulator(
                str(selected[(tag, seed)]), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
            data[value][seed] = {
                scalar_tag: np.asarray(
                    [event.value for event in accumulator.Scalars(scalar_tag)],
                    dtype=float,
                )
                for scalar_tag in TAGS
            }
    return data


def rolling_mean(values, window=WINDOW):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    ci = float(T_CRITICAL_95_DF4 * values.std(ddof=1) / np.sqrt(values.size))
    return mean, ci


def parameter_colors(count):
    return plt.colormaps["viridis"](np.linspace(0.08, 0.92, count))


def parameter_labels(task):
    symbol = SWEEPS[task]["symbol"]
    return [f"{symbol}={display_value(value)}" for value in SWEEPS[task]["values"]]


def plot_learning_curves(ax, task, data, colors):
    for color, label, value in zip(
        colors, parameter_labels(task), SWEEPS[task]["values"]
    ):
        curves = np.stack(
            [rolling_mean(data[value][seed]["Episode_Return"]) for seed in SEEDS]
        )
        mean = curves.mean(axis=0)
        ci = T_CRITICAL_95_DF4 * curves.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        episodes = np.arange(WINDOW, WINDOW + mean.size)
        ax.fill_between(episodes, mean - ci, mean + ci, color=color, alpha=0.12, linewidth=0)
        ax.plot(episodes, mean, color=color, linewidth=1.8, label=label)
    style_axis(ax, "Episode", "Episode return")
    ax.set_xlim(WINDOW, 1000)
    ax.set_title("(a) Learning curves", loc="left")


def plot_seed_summary(ax, task, per_value, ylabel, panel, colors, value_format=".1f"):
    values = SWEEPS[task]["values"]
    x = np.arange(len(values))
    offsets = np.linspace(-0.13, 0.13, len(SEEDS))
    for index, (color, parameter_value) in enumerate(zip(colors, values)):
        samples = np.asarray(per_value[parameter_value], dtype=float)
        mean, ci = mean_ci(samples)
        ax.scatter(
            index + offsets,
            samples,
            s=23,
            color=color,
            alpha=0.40,
            edgecolors="none",
            zorder=2,
        )
        ax.errorbar(
            index,
            mean,
            yerr=ci,
            fmt="o",
            color=color,
            markersize=7,
            capsize=4,
            linewidth=1.8,
            zorder=3,
        )
        ax.annotate(
            format(mean, value_format),
            (index, mean),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )
    ax.set_xticks(x, [display_value(value) for value in values])
    style_axis(ax, SWEEPS[task]["symbol"], ylabel)
    ax.set_title(panel, loc="left")


def performance_figure(task, data):
    colors = parameter_colors(len(SWEEPS[task]["values"]))
    fig, (ax_learning, ax_final) = plt.subplots(2, 1, figsize=(9.0, 8.6))
    plot_learning_curves(ax_learning, task, data, colors)

    final_returns = {
        value: [
            data[value][seed]["Episode_Return"][-FINAL_EPISODES:].mean()
            for seed in SEEDS
        ]
        for value in SWEEPS[task]["values"]
    }
    plot_seed_summary(
        ax_final,
        task,
        final_returns,
        "Mean return, episodes 901-1000",
        "(b) Final performance",
        colors,
    )

    handles, labels = ax_learning.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.012,
        "Lines and points are means across five seeds; shading and error bars are 95% confidence intervals. "
        "Learning curves use a 50-episode moving average.",
        ha="center",
        fontsize=8.2,
    )
    fig.subplots_adjust(top=0.89, bottom=0.10, left=0.11, right=0.98, hspace=0.43)
    save_figure(fig, f"task2-{task}-{SWEEPS[task]['parameter']}-performance", formats=("png", "svg", "pdf"))


def time_to_threshold(returns):
    curve = rolling_mean(returns)
    reached = np.flatnonzero(curve >= PERFORMANCE_THRESHOLD)
    return int(reached[0] + WINDOW) if reached.size else 1000


def support_figure(task, data):
    colors = parameter_colors(len(SWEEPS[task]["values"]))
    fig, (ax_speed, ax_support) = plt.subplots(2, 1, figsize=(9.0, 8.3))

    threshold_times = {
        value: [
            time_to_threshold(data[value][seed]["Episode_Return"])
            for seed in SEEDS
        ]
        for value in SWEEPS[task]["values"]
    }
    plot_seed_summary(
        ax_speed,
        task,
        threshold_times,
        "Episode reaching return threshold",
        r"(a) Time to 50-episode mean return $\geq -20$ (lower is better)",
        colors,
        value_format=".0f",
    )
    ax_speed.set_ylim(bottom=0, top=1040)

    if task == "q2":
        supporting_metric = {
            value: [
                data[value][seed]["Cliff_Falls"][:EARLY_EPISODES].mean()
                for seed in SEEDS
            ]
            for value in SWEEPS[task]["values"]
        }
        ylabel = "Cliff falls per episode"
        panel = "(b) Early-training cliff falls, episodes 1-300 (lower is better)"
        value_format = ".2f"
        note = (
            "Threshold times not achieved by episode 1000 are plotted at 1000. "
            "Early cliff-fall rates are calculated over episodes 1-300."
        )
    else:
        supporting_metric = {
            value: [
                100.0
                * np.mean(data[value][seed]["Episode_Length"][:EARLY_EPISODES] >= 200)
                for seed in SEEDS
            ]
            for value in SWEEPS[task]["values"]
        }
        ylabel = "Episodes at least 200 steps (%)"
        panel = "(b) Long-episode rate, episodes 1-300 (lower is better)"
        value_format = ".1f"
        note = (
            "Threshold times not achieved by episode 1000 are plotted at 1000. "
            "For lambda=1.00, training used a 200-step safety cap after uncapped runs failed to terminate promptly."
        )

    plot_seed_summary(
        ax_support,
        task,
        supporting_metric,
        ylabel,
        panel,
        colors,
        value_format=value_format,
    )
    ax_support.set_ylim(bottom=0)
    fig.text(0.5, 0.012, note, ha="center", fontsize=8.2)
    fig.subplots_adjust(top=0.96, bottom=0.10, left=0.11, right=0.98, hspace=0.43)
    stem = "convergence-safety" if task == "q2" else "convergence-stability"
    save_figure(
        fig,
        f"task2-{task}-{SWEEPS[task]['parameter']}-{stem}",
        formats=("png", "svg", "pdf"),
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for task in ("q2", "q3"):
        data = load_sweep(task)
        performance_figure(task, data)
        support_figure(task, data)


if __name__ == "__main__":
    main()
