from plot_tensorboard_results import (
    COLORS,
    LINESTYLES,
    LOG_ROOT,
    RESULTS_DIR,
    RUN_ORDER,
    save_figure,
    style_axis,
)

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SEEDS = range(5)
RUN_MARKERS = {
    "Q-learning": "task2_qlearning",
    "SARSA(0)": "task2_sarsa0",
    "SARSA(5)": "task2_sarsa5",
    "SARSA(lambda=0.1)": "task2_lambda01",
}
T_CRITICAL_95_DF4 = 2.776


def select_runs():
    selected = {label: {} for label in RUN_ORDER}
    for directory in LOG_ROOT.iterdir():
        if not directory.is_dir():
            continue
        for label, marker in RUN_MARKERS.items():
            for seed in SEEDS:
                if f"{marker}_seed{seed}_" not in directory.name:
                    continue
                current = selected[label].get(seed)
                if current is None or directory.stat().st_mtime > current.stat().st_mtime:
                    selected[label][seed] = directory

    missing = [
        f"{label}, seed {seed}"
        for label in RUN_ORDER
        for seed in SEEDS
        if seed not in selected[label]
    ]
    if missing:
        raise RuntimeError("Missing runs: " + ", ".join(missing))
    return selected


def load_runs(run_dirs):
    data = {label: {} for label in RUN_ORDER}
    q_tables = {label: {} for label in RUN_ORDER}
    for label in RUN_ORDER:
        for seed in SEEDS:
            directory = run_dirs[label][seed]
            accumulator = EventAccumulator(
                str(directory), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
            data[label][seed] = {}
            for tag in ("Episode_Return", "Episode_Length", "Cliff_Falls", "TD_Error"):
                events = accumulator.Scalars(tag)
                data[label][seed][tag] = np.asarray(
                    [event.value for event in events], dtype=float
                )
            q_tables[label][seed] = np.load(directory / "q_table.npz")["q_table"]
    return data, q_tables


def rolling_mean(values, window=50):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def mean_and_ci(data, label, tag, window=50):
    series = np.stack(
        [rolling_mean(data[label][seed][tag], window) for seed in SEEDS]
    )
    mean = series.mean(axis=0)
    ci = T_CRITICAL_95_DF4 * series.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    steps = np.arange(window - 1, window - 1 + len(mean))
    return steps, mean, ci


def plot_mean_ci(ax, data, tag, window=50):
    for label in RUN_ORDER:
        steps, mean, ci = mean_and_ci(data, label, tag, window)
        ax.fill_between(
            steps,
            mean - ci,
            mean + ci,
            color=COLORS[label],
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(
            steps,
            mean,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            linewidth=2.0,
            label=label,
        )


def final_return_stats(data):
    stats = {}
    for label in RUN_ORDER:
        per_seed = np.asarray(
            [data[label][seed]["Episode_Return"][-100:].mean() for seed in SEEDS]
        )
        mean = float(per_seed.mean())
        ci = float(T_CRITICAL_95_DF4 * per_seed.std(ddof=1) / np.sqrt(len(SEEDS)))
        stats[label] = (per_seed, mean, ci)
    return stats


def performance_figure(data):
    fig, (ax_curve, ax_final) = plt.subplots(
        1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": (1.75, 1.0)}
    )

    plot_mean_ci(ax_curve, data, "Episode_Return", window=50)
    style_axis(ax_curve, "Episode", "Episode return")
    ax_curve.set_xlim(49, 650)
    ax_curve.set_ylim(-450, 0)
    ax_curve.set_title("(a) Learning curves", loc="left")

    stats = final_return_stats(data)
    y_positions = np.arange(len(RUN_ORDER))
    seed_offsets = np.linspace(-0.13, 0.13, len(SEEDS))
    for y, label in zip(y_positions, RUN_ORDER):
        per_seed, mean, ci = stats[label]
        ax_final.scatter(
            per_seed,
            y + seed_offsets,
            s=20,
            color=COLORS[label],
            alpha=0.42,
            edgecolors="none",
        )
        ax_final.errorbar(
            mean,
            y,
            xerr=ci,
            fmt="o",
            markersize=7,
            capsize=4,
            color=COLORS[label],
            linewidth=2.0,
        )
        ax_final.text(mean + 0.2, y, f"{mean:.1f}", va="center", fontsize=9)
    ax_final.set_yticks(y_positions, RUN_ORDER)
    ax_final.invert_yaxis()
    ax_final.set_xlim(-22.5, -11.5)
    style_axis(ax_final, "Mean return, episodes 901-1000", "")
    ax_final.set_title("(b) Final performance", loc="left")

    handles, labels = ax_curve.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "Lines are means across five seeds; shading and error bars are 95% confidence intervals. "
        "Curves use a 50-episode moving average.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.08, right=0.98, wspace=0.36)
    save_figure(
        fig,
        "task2-q1-multiseed-performance",
        formats=("png", "svg", "pdf"),
    )


def safety_efficiency_figure(data):
    fig, (ax_falls, ax_length) = plt.subplots(1, 2, figsize=(11.2, 4.35))

    plot_mean_ci(ax_falls, data, "Cliff_Falls", window=50)
    style_axis(ax_falls, "Episode", "Cliff falls per episode")
    ax_falls.set_xlim(49, 650)
    ax_falls.set_ylim(bottom=0)
    ax_falls.set_title("(a) Cliff falls", loc="left")

    plot_mean_ci(ax_length, data, "Episode_Length", window=50)
    style_axis(ax_length, "Episode", "Steps per episode")
    ax_length.set_xlim(49, 650)
    ax_length.set_ylim(bottom=0)
    ax_length.set_title("(b) Episode length", loc="left")

    handles, labels = ax_falls.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "Means across five seeds with 95% confidence intervals; 50-episode moving averages.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.08, right=0.98, wspace=0.28)
    save_figure(
        fig,
        "task2-q1-safety-efficiency",
        formats=("png", "svg", "pdf"),
    )


def greedy_trajectory(q_table, max_steps=100):
    state = 36
    path = [state]
    for _ in range(max_steps):
        action = int(np.argmax(q_table[state]))
        row, col = divmod(state, 12)
        if action == 0:
            row = max(row - 1, 0)
        elif action == 1:
            col = min(col + 1, 11)
        elif action == 2:
            row = min(row + 1, 3)
        else:
            col = max(col - 1, 0)

        if row == 3 and 1 <= col <= 10:
            state = 36
        else:
            state = row * 12 + col
        path.append(state)
        if state == 47:
            break
    return path


def draw_grid(ax):
    cliff_color = "#F3C1C1"
    start_color = "#DCEED8"
    goal_color = "#D7EAF5"
    for row in range(4):
        for col in range(12):
            facecolor = "#FAFAFA"
            if row == 3 and 1 <= col <= 10:
                facecolor = cliff_color
            elif (row, col) == (3, 0):
                facecolor = start_color
            elif (row, col) == (3, 11):
                facecolor = goal_color
            rectangle = plt.Rectangle(
                (col, row), 1, 1, facecolor=facecolor, edgecolor="#B8B8B8", linewidth=0.7
            )
            ax.add_patch(rectangle)
    ax.text(0.5, 3.5, "S", ha="center", va="center", fontweight="bold")
    ax.text(11.5, 3.5, "G", ha="center", va="center", fontweight="bold")
    ax.text(5.5, 3.5, "CLIFF", ha="center", va="center", fontsize=8, color="#8A3B3B")
    ax.set_xlim(0, 12)
    ax.set_ylim(4, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def policy_paths_figure(q_tables):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 4.75))
    for panel, ax, label in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat, RUN_ORDER):
        draw_grid(ax)
        paths = [greedy_trajectory(q_tables[label][seed]) for seed in SEEDS]
        lengths = np.asarray([len(path) - 1 for path in paths])
        median_length = np.median(lengths)
        representative = int(np.argmin(np.abs(lengths - median_length)))

        for seed, path in zip(SEEDS, paths):
            coordinates = np.asarray([(state % 12 + 0.5, state // 12 + 0.5) for state in path])
            ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color=COLORS[label],
                linewidth=1.6,
                alpha=0.22,
                marker="o",
                markersize=2.5,
            )

        path = paths[representative]
        coordinates = np.asarray([(state % 12 + 0.5, state // 12 + 0.5) for state in path])
        ax.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color=COLORS[label],
            linewidth=2.7,
            marker="o",
            markersize=3.2,
        )
        ax.set_title(
            f"{panel} {label}: greedy path length {lengths.mean():.1f} +/- {lengths.std(ddof=1):.1f}",
            loc="left",
            fontsize=10,
        )

    fig.text(
        0.5,
        0.02,
        "Bold line: representative median-length path; faint lines: all seeds. "
        "Q-learning follows the cliff edge, while SARSA variants learn safer routes.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.96, bottom=0.12, left=0.04, right=0.98, hspace=0.42, wspace=0.12)
    save_figure(
        fig,
        "task2-q1-learned-greedy-paths",
        formats=("png", "svg", "pdf"),
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = select_runs()
    data, q_tables = load_runs(run_dirs)
    performance_figure(data)
    safety_efficiency_figure(data)
    policy_paths_figure(q_tables)
    print("Created Task 2 Q1 figures from 20 TensorBoard runs and their saved Q-tables.")


if __name__ == "__main__":
    main()
