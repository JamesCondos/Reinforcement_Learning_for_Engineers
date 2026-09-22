"""Export the Workshop 3 Task 1 Question 2 evaluation curve."""

from pathlib import Path
import csv
import os


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "Results" / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN = ROOT / "logs" / "CartPole-v1" / "DQN_22-09-2026_21-25-23"
OUTPUT = ROOT / "Results"
CSV = OUTPUT / "task1-q2-cartpole-evaluation.csv"


def main():
    if RUN.exists():
        events = EventAccumulator(str(RUN), size_guidance={"scalars": 0})
        events.Reload()
        evaluations = events.Scalars("Eval_Returns")
        steps = np.array([entry.step for entry in evaluations], dtype=int)
        returns = np.array([entry.value for entry in evaluations], dtype=float)
    else:
        with CSV.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        steps = np.array([int(row["environment_steps"]) for row in rows])
        returns = np.array([float(row["evaluation_return"]) for row in rows])

    if len(steps) == 0:
        raise ValueError("No evaluation returns were found in the TensorBoard run")

    window = 5
    rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
    rolling_steps = steps[window - 1 :]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.25), layout="constrained")

    ax.plot(
        steps,
        returns,
        color="#81A9C5",
        linewidth=1.1,
        marker="o",
        markersize=2.1,
        alpha=0.85,
        label="Evaluation return",
        zorder=2,
    )
    ax.plot(
        rolling_steps,
        rolling,
        color="#123D62",
        linewidth=2.0,
        label="5-evaluation rolling mean",
        zorder=3,
    )
    ax.axhline(500, color="#7B858C", linewidth=0.9, linestyle=(0, (4, 3)), label="Maximum return (500)")

    ax.set(xlabel="Environment steps", ylabel="Evaluation return")
    ax.set_xlim(0, 170_000)
    ax.set_ylim(0, 525)
    ax.xaxis.set_major_locator(MultipleLocator(25_000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k" if value else "0"))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.grid(axis="y", color="#DCE2E6", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=True, edgecolor="#DCE2E6", facecolor="white", fontsize=8)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "task1-q2-cartpole-evaluation"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)

    with CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["environment_steps", "evaluation_return"])
        writer.writerows(zip(steps, returns))

    print(f"Exported {len(steps)} evaluations to {OUTPUT}")
    print(f"Final return: {returns[-1]:.0f}; maximum return: {returns.max():.0f}")


if __name__ == "__main__":
    main()
