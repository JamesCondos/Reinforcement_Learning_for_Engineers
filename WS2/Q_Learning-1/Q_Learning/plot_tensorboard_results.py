import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_ROOT = HERE / "logs" / "CliffWalking-v1"
RESULTS_DIR = HERE.parents[1] / "Results"
MATPLOTLIB_CACHE = HERE.parents[1] / "tmp" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 12,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

RUN_ORDER = ["Q-learning", "SARSA(0)", "SARSA(5)", "SARSA(lambda=0.1)"]
COLORS = {
    "Q-learning": "#0072B2",
    "SARSA(0)": "#E69F00",
    "SARSA(5)": "#009E73",
    "SARSA(lambda=0.1)": "#CC79A7",
}
LINESTYLES = {
    "Q-learning": "-",
    "SARSA(0)": "--",
    "SARSA(5)": "-.",
    "SARSA(lambda=0.1)": ":",
}
TAG_NAMES = {
    "Episode Return": "Episode_Return",
    "Epsilon-greedy value": "Epsilon-Greedy_Value",
    "Reward": "Reward",
    "TD error": "TD_Error",
}


def classify_run(name):
    if name.startswith("QLearning_"):
        return "Q-learning"
    if name.startswith("SARSA_step5_"):
        return "SARSA(5)"
    if name.startswith("SARSA_lambda0.1_"):
        return "SARSA(lambda=0.1)"
    if name.startswith("SARSA_"):
        return "SARSA(0)"
    return None


def latest_runs():
    selected = {}
    for directory in LOG_ROOT.iterdir():
        if not directory.is_dir():
            continue
        label = classify_run(directory.name)
        if label is None:
            continue
        if label not in selected or directory.stat().st_mtime > selected[label].stat().st_mtime:
            selected[label] = directory

    missing = [label for label in RUN_ORDER if label not in selected]
    if missing:
        raise RuntimeError(f"Missing TensorBoard runs: {', '.join(missing)}")
    return selected


def load_scalars(run_dirs):
    data = {}
    for label in RUN_ORDER:
        accumulator = EventAccumulator(
            str(run_dirs[label]), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        data[label] = {}
        for display_name, tag in TAG_NAMES.items():
            events = accumulator.Scalars(tag)
            data[label][display_name] = (
                np.asarray([event.step for event in events], dtype=float),
                np.asarray([event.value for event in events], dtype=float),
            )
    return data


def moving_average(steps, values, window, absolute=False):
    if absolute:
        values = np.abs(values)
    kernel = np.ones(window, dtype=float) / window
    return steps[window - 1 :], np.convolve(values, kernel, mode="valid")


def style_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def plot_series(ax, data, metric, window=None, absolute=False):
    for label in RUN_ORDER:
        steps, values = data[label][metric]
        if window is not None:
            steps, values = moving_average(steps, values, window, absolute=absolute)
        ax.plot(
            steps,
            values,
            label=label,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            linewidth=1.9,
        )


def plot_epsilon_schedule(ax, data):
    steps, values = data["Q-learning"]["Epsilon-greedy value"]
    ax.plot(
        steps,
        values,
        color="#333333",
        linewidth=2.0,
        label="All algorithms (identical schedule)",
    )


def save_figure(fig, stem, formats=("png", "svg")):
    for extension in formats:
        kwargs = {"dpi": 350} if extension == "png" else {}
        fig.savefig(RESULTS_DIR / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def episode_return_figure(data):
    fig, (ax_full, ax_late) = plt.subplots(1, 2, figsize=(12.2, 4.3))
    plot_series(ax_full, data, "Episode Return", window=50)
    style_axis(ax_full, "Episode", "Return")
    ax_full.set_ylim(-400, 0)
    ax_full.set_title("Learning phase")

    plot_series(ax_late, data, "Episode Return", window=50)
    style_axis(ax_late, "Episode", "Return")
    ax_late.set_xlim(400, 999)
    ax_late.set_ylim(-20, -10)
    ax_late.set_title("Final-performance detail")

    handles, labels = ax_full.get_legend_handles_labels()
    fig.suptitle("CliffWalking-v1: Episode Return (50-episode moving average)", y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, "task1-q2-episode-return")


def metric_figure(data, metric, stem, xlabel, ylabel, window=None, absolute=False):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    plot_series(ax, data, metric, window=window, absolute=absolute)
    style_axis(ax, xlabel, ylabel)
    title = f"CliffWalking-v1: {metric}"
    if absolute:
        title = "CliffWalking-v1: Mean absolute TD error"
    if window is not None:
        title += f" ({window}-step moving average)"
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, stem)


def epsilon_figure(data):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    plot_epsilon_schedule(ax, data)
    style_axis(ax, "Episode", "Epsilon")
    ax.set_title("CliffWalking-v1: Epsilon-greedy exploration schedule")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_figure(fig, "epsilon-greedy-value")


def combined_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
    specifications = [
        ("Episode Return", "Episode", "Return", 50, False),
        ("Epsilon-greedy value", "Episode", "Epsilon", None, False),
        ("Reward", "Environment step", "Mean immediate reward", 500, False),
        ("TD error", "Environment step", "Mean absolute TD error", 500, True),
    ]
    for ax, (metric, xlabel, ylabel, window, absolute) in zip(axes.flat, specifications):
        if metric == "Epsilon-greedy value":
            plot_epsilon_schedule(ax, data)
        else:
            plot_series(ax, data, metric, window=window, absolute=absolute)
        style_axis(ax, xlabel, ylabel)
        if metric == "TD error":
            title = f"Mean absolute TD error ({window}-step mean)"
        elif window is not None:
            title = f"{metric} ({window}-step mean)"
        else:
            title = "Epsilon-greedy schedule"
        ax.set_title(title)
        if metric == "Episode Return":
            ax.set_ylim(-400, 0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("CliffWalking-v1: TensorBoard Training Metrics", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, "cliffwalking-all-training-metrics", formats=("png", "pdf"))


def task2_q1_metrics(data, window=50, tolerance=5.0):
    metrics = {}
    for label in RUN_ORDER:
        steps, returns = data[label]["Episode Return"]
        smooth_steps, smooth_returns = moving_average(steps, returns, window)
        final_mean = float(np.mean(returns[-100:]))
        outside = np.abs(smooth_returns - final_mean) > tolerance
        last_outside = np.flatnonzero(outside)
        if len(last_outside) == 0:
            convergence_episode = int(smooth_steps[0])
        elif last_outside[-1] + 1 < len(smooth_steps):
            convergence_episode = int(smooth_steps[last_outside[-1] + 1])
        else:
            convergence_episode = None
        metrics[label] = {
            "final_mean": final_mean,
            "convergence_episode": convergence_episode,
        }
    return metrics


def task2_q1_figure(data):
    metrics = task2_q1_metrics(data)
    fig = plt.figure(figsize=(10.2, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.45, 1.0), hspace=0.48, wspace=0.38)
    ax_curve = fig.add_subplot(grid[0, :])
    ax_speed = fig.add_subplot(grid[1, 0])
    ax_final = fig.add_subplot(grid[1, 1])

    plot_series(ax_curve, data, "Episode Return", window=50)
    style_axis(ax_curve, "Episode", "Episode return")
    ax_curve.set_xlim(49, 600)
    ax_curve.set_ylim(-400, 0)
    ax_curve.set_title("(a) Learning curves: 50-episode moving average", loc="left")

    y_positions = np.arange(len(RUN_ORDER))
    convergence = [metrics[label]["convergence_episode"] for label in RUN_ORDER]
    for y, label, value in zip(y_positions, RUN_ORDER, convergence):
        ax_speed.barh(y, value, color=COLORS[label], height=0.55, alpha=0.9)
        ax_speed.text(value + 8, y, f"{value}", va="center", ha="left", fontsize=9)
    ax_speed.set_yticks(y_positions, RUN_ORDER)
    ax_speed.invert_yaxis()
    ax_speed.set_xlim(0, max(convergence) * 1.18)
    style_axis(ax_speed, "Episode", "")
    ax_speed.set_title("(b) Convergence episode (lower is faster)", loc="left")

    final_means = [metrics[label]["final_mean"] for label in RUN_ORDER]
    for y, label, value in zip(y_positions, RUN_ORDER, final_means):
        ax_final.hlines(y, -19, value, color=COLORS[label], linewidth=2.2)
        ax_final.plot(value, y, "o", color=COLORS[label], markersize=7)
        ax_final.text(value + 0.22, y, f"{value:.0f}", va="center", ha="left", fontsize=9)
    ax_final.set_yticks(y_positions, RUN_ORDER)
    ax_final.invert_yaxis()
    ax_final.set_xlim(-19, -11.5)
    style_axis(ax_final, "Mean return (episodes 901-1000)", "")
    ax_final.set_title("(c) Final performance (higher is better)", loc="left")

    handles, labels = ax_curve.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "Single run per algorithm. Convergence is the first episode after which the 50-episode "
        "mean stays within +/-5 reward of the final 100-episode mean.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.11, right=0.98)
    save_figure(
        fig,
        "task2-q1-performance-comparison",
        formats=("png", "svg", "pdf"),
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = latest_runs()
    data = load_scalars(run_dirs)

    episode_return_figure(data)
    epsilon_figure(data)
    metric_figure(
        data,
        "Reward",
        "reward-per-environment-step",
        "Environment step",
        "Immediate reward",
        window=500,
    )
    metric_figure(
        data,
        "TD error",
        "td-error-per-environment-step",
        "Environment step",
        "Mean absolute TD error",
        window=500,
        absolute=True,
    )
    combined_figure(data)
    task2_q1_figure(data)

    print("Runs used:")
    for label in RUN_ORDER:
        print(f"  {label}: {run_dirs[label].name}")
    print(f"Figures saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
