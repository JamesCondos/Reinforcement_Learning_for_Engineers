"""Run the Task 2 Q2/Q3 parameter sweeps without generating figures."""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EPISODES = 1000
SEEDS = range(5)
LEARNING_RATES = (0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 1.00)
LAMBDA_VALUES = (0.00, 0.10, 0.30, 0.50, 0.70, 0.90, 1.00)
LOG_ROOT = Path("logs") / "CliffWalking-v1"


def value_tag(value):
    return f"{value:.2f}".replace(".", "p")


def run_experiment(experiment_name, seed, extra_args):
    command = [
        sys.executable,
        "run_q_learning.py",
        "--env_name",
        "CliffWalking-v1",
        "--episodes",
        str(EPISODES),
        "--max_episode_steps",
        "200",
        "--seed",
        str(seed),
        "--save_qtable",
        "--exp_name",
        experiment_name,
        *extra_args,
    ]
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    print(f"Completed {experiment_name} in {elapsed:.1f} s", flush=True)


def has_completed(experiment_name):
    return any(
        (directory / "q_table.npz").is_file()
        for directory in LOG_ROOT.glob(f"*_{experiment_name}_*")
        if directory.is_dir()
    )


def main():
    experiments = []
    for alpha in LEARNING_RATES:
        for seed in SEEDS:
            name = f"task2_q2_alpha{value_tag(alpha)}_seed{seed}"
            experiments.append(
                (name, seed, ["--off_policy", "--learning_rate", str(alpha)])
            )

    for trace_lambda in LAMBDA_VALUES:
        for seed in SEEDS:
            name = f"task2_q3_lambda{value_tag(trace_lambda)}_seed{seed}"
            lambda_args = (
                []
                if trace_lambda == 0.0
                else ["--sarsa_lambda", str(trace_lambda)]
            )
            experiments.append((name, seed, lambda_args))

    total = len(experiments)
    pending = []
    for name, seed, extra_args in experiments:
        if has_completed(name):
            print(f"Skipping completed {name}", flush=True)
        else:
            pending.append((name, seed, extra_args))

    print(f"Launching {len(pending)} pending runs with five workers.", flush=True)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(run_experiment, name, seed, extra_args): name
            for name, seed, extra_args in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            future.result()
            print(f"Progress: {completed}/{len(pending)} pending runs ({name})", flush=True)

    print(f"Finished all {total} Task 2 parameter-sweep runs.", flush=True)


if __name__ == "__main__":
    main()
