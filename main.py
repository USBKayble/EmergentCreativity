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

    nn_agent = None
    if args.nn:
        try:
            import torch
            from src.emergent_creativity.nn.architecture import TenantNetwork
            from src.emergent_creativity.environment.senses import TOTAL_SENSORY_DIM
            VITALS_DIM = 4
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = TenantNetwork()
            ck  = torch.load(args.nn, map_location=device)
            net.load_state_dict(ck["model"])
            net.to(device)
            net.eval()
            lstm_state = [None]  # mutable container

            def nn_agent(obs, _):
                import numpy as np
                vis = torch.from_numpy(obs["vision"]).float()
                vis = vis.permute(2, 0, 1).unsqueeze(0).to(device)
                nv  = np.concatenate([
                    obs["hearing"], obs["touch"], obs["smell"],
                    obs["taste"],  obs["vitals"],
                ])
                nv  = torch.from_numpy(nv).float().unsqueeze(0).to(device)
                action, _, _, new_state = net.get_action(vis, nv, lstm_state[0])
                lstm_state[0] = new_state
                return action

            print(f"[main] NN agent loaded from {args.nn}")
        except Exception as e:
            print(f"[main] Could not load NN agent: {e}. Falling back to manual mode.")

    print("[main] Starting viewer … (close window or press Q to exit)")
    env = TenantEnv(gui=args.gui)
    viewer = SimViewer(env, nn_agent=nn_agent, target_fps=args.fps)
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
    vp.add_argument("--nn",  type=str,  default=None, help="Path to NN checkpoint")
    vp.add_argument("--gui", action="store_true",     help="Use PyBullet GUI window")
    vp.add_argument("--fps", type=int,  default=30,   help="Target FPS")

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
