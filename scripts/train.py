"""
scripts/train.py
================
Console entry-point for RL training.

Usage::

    python -m scripts.train --steps 500000 --lr 0.0003
    # or via project entry-point:
    ec-train --steps 500000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the EmergentCreativity NN agent")
    parser.add_argument("--steps",    type=int,   default=1_000_000)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--n-steps",  type=int,   default=2048,  dest="n_steps")
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--save-dir", type=str,   default="checkpoints", dest="save_dir")
    parser.add_argument("--log-dir",  type=str,   default="logs/tensorboard", dest="log_dir")
    parser.add_argument("--resume",   type=str,   default=None)
    parser.add_argument("--config",   type=str,   default=None,
                        help="Path to rewards.yaml override")
    args = parser.parse_args()

    from src.emergent_creativity.sim_env import TenantEnv
    from src.emergent_creativity.nn.trainer import PPOTrainer

    env = TenantEnv(gui=False, config_path=args.config)
    trainer = PPOTrainer(
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
    if args.resume:
        trainer.load(args.resume)

    trainer.train(total_timesteps=args.steps)


if __name__ == "__main__":
    main()
