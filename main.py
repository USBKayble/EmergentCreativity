"""
main.py
=======
EmergentCreativity – quick-start entry point.

Run modes
---------
    python main.py view          – launch the interactive viewer (manual control)
    python main.py train         – start RL training
    python main.py view --nn <checkpoint.pt>  – view with trained NN agent

Examples
--------
::

    # Interactive manual control (no GPU needed)
    python main.py view

    # Train the neural network (GPU recommended)
    python main.py train --steps 1000000

    # Watch a trained agent
    python main.py view --nn checkpoints/checkpoint_100000.pt
"""
from __future__ import annotations

import argparse
import sys


def cmd_view(args: argparse.Namespace) -> None:
    from src.emergent_creativity.sim_env import TenantEnv
    from src.emergent_creativity.ui.viewer import SimViewer

    online_learner = None
    nn_agent       = None

    try:
        from src.emergent_creativity.nn.online_learner import OnlineLearner
        online_learner = OnlineLearner(
            n_steps=128,
            device="auto",
            save_dir=args.save_dir if hasattr(args, "save_dir") else "checkpoints",
            save_freq=5000,
        )
        if args.nn:
            try:
                online_learner.load(args.nn)
                print(f"[main] OnlineLearner loaded from {args.nn}")
            except Exception as e:
                print(f"[main] Could not load checkpoint {args.nn}: {e}")
        else:
            print("[main] OnlineLearner starting fresh — will learn from scratch.")
    except ImportError as e:
        print(f"[main] PyTorch unavailable — falling back to manual mode ({e})")

    print("[main] Starting viewer … (close window or press Q to exit)")
    env = TenantEnv(gui=args.gui)
    viewer = SimViewer(
        env,
        nn_agent=nn_agent,
        online_learner=online_learner,
        target_fps=args.fps,
    )
    viewer.run()


def cmd_train(args: argparse.Namespace) -> None:
    from src.emergent_creativity.sim_env import TenantEnv
    from src.emergent_creativity.nn.trainer import PPOTrainer

    print(f"[main] Training for {args.steps:,} steps …")
    env    = TenantEnv(gui=False)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="emergent-creativity",
        description="EmergentCreativity – 3D apartment RL simulation",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- view ----
    vp = sub.add_parser("view", help="Launch interactive viewer")
    vp.add_argument("--nn",       type=str,  default=None, help="Path to checkpoint to resume from")
    vp.add_argument("--gui",      action="store_true",     help="Use PyBullet GUI window")
    vp.add_argument("--fps",      type=int,  default=30,   help="Target FPS")
    vp.add_argument("--save-dir", type=str,  default="checkpoints", help="Auto-checkpoint directory")

    # ---- train ----
    tp = sub.add_parser("train", help="Train the NN agent")
    tp.add_argument("--steps",    type=int,   default=1_000_000, help="Total timesteps")
    tp.add_argument("--lr",       type=float, default=3e-4,      help="Learning rate")
    tp.add_argument("--n-steps",  type=int,   default=2048,      help="Rollout steps")
    tp.add_argument("--batch",    type=int,   default=64,        help="Mini-batch size")
    tp.add_argument("--save-dir", type=str,   default="checkpoints")
    tp.add_argument("--log-dir",  type=str,   default="logs/tensorboard")
    tp.add_argument("--resume",   type=str,   default=None, help="Checkpoint to resume from")

    args = parser.parse_args()

    if args.command == "view":
        cmd_view(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
