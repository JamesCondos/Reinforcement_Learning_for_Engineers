import subprocess
import sys


ALGORITHMS = {
    "qlearning": ["--off_policy"],
    "sarsa0": [],
    "sarsa5": ["--look_ahead_step", "5"],
    "lambda01": ["--sarsa_lambda", "0.1"],
}


def main():
    for seed in range(5):
        for name, extra_args in ALGORITHMS.items():
            command = [
                sys.executable,
                "run_q_learning.py",
                "--env_name",
                "CliffWalking-v1",
                "--episodes",
                "1000",
                "--seed",
                str(seed),
                "--save_qtable",
                "--exp_name",
                f"task2_{name}_seed{seed}",
                *extra_args,
            ]
            print(f"Running {name}, seed {seed}...", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
