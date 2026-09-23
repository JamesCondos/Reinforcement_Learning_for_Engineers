import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    ("learning_rate/lr1e-4", ["--learning_rate", "0.0001", "--exp_name", "lr1e-4"]),
    ("learning_rate/lr1e-3", ["--learning_rate", "0.001", "--exp_name", "lr1e-3"]),
    ("learning_rate/lr1e-2", ["--learning_rate", "0.01", "--exp_name", "lr1e-2"]),
    ("exploration/const_eps_0.1", ["--epsilon", "0.1", "--epsilon_decay", "1.0", "--exp_name", "const_eps_0.1"]),
    ("exploration/eps_decay_0.9", ["--epsilon", "1.0", "--epsilon_decay", "0.9", "--epsilon_min", "0.02", "--exp_name", "0.9t"]),
    ("exploration/eps_decay_0.995", ["--epsilon", "1.0", "--epsilon_decay", "0.995", "--epsilon_min", "0.02", "--exp_name", "0.995t"]),
    ("replay_buffer/buffer1e5", ["--memory_size", "100000", "--exp_name", "buffer1e5"]),
    ("replay_buffer/buffer1e4", ["--memory_size", "10000", "--exp_name", "buffer1e4"]),
    ("replay_buffer/buffer1e3", ["--memory_size", "1000", "--exp_name", "buffer1e3"]),
    ("batch_size/batch128", ["--batch_size", "128", "--exp_name", "batch128"]),
    ("batch_size/batch32", ["--batch_size", "32", "--exp_name", "batch32"]),
]


def main():
    root = Path(__file__).resolve().parent
    result_root = root / "Results" / "task2"
    for name, arguments in EXPERIMENTS:
        result_dir = result_root / name
        if (result_dir / "config.json").exists():
            print(f"\n=== Skipping completed {name} ===", flush=True)
            continue
        result_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(root / "DQN" / "run_dqn.py"),
            "--env_name", "CartPole-v1",
            "--results_dir", str(result_dir),
            "--device", "cpu",
            "--no_tensorboard",
            *arguments,
        ]
        print(f"\n=== Running {name} ===", flush=True)
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
