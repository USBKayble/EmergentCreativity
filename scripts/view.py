"""
scripts/view.py
===============
Console entry-point for the interactive viewer with online learning.

Usage::

    python -m scripts.view                             # fresh online learning (random start)
    python -m scripts.view --nn checkpoints/online_5000.pt  # resume from checkpoint
    # or via project entry-point:
    ec-view
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="View the EmergentCreativity simulation")
    parser.add_argument("--nn",       type=str, default=None,
                        help="Checkpoint to load (optional; starts fresh if omitted)")
    parser.add_argument("--gui",      action="store_true", help="Open PyBullet GUI")
    parser.add_argument("--fps",      type=int, default=30)
    parser.add_argument("--config",   type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory for auto-saved checkpoints")
    args = parser.parse_args()

    from src.emergent_creativity.sim_env import TenantEnv
    from src.emergent_creativity.ui.viewer import SimViewer

    online_learner = None
    try:
        from src.emergent_creativity.nn.online_learner import OnlineLearner
        online_learner = OnlineLearner(
            n_steps=128,
            device="auto",
            save_dir=args.save_dir,
            save_freq=5000,
        )
        if args.nn:
            online_learner.load(args.nn)
        else:
            print("[view] No checkpoint given — OnlineLearner starting from scratch.")
    except ImportError as e:
        print(f"[view] PyTorch unavailable — falling back to manual mode ({e})")

    env    = TenantEnv(gui=args.gui, config_path=args.config)
    viewer = SimViewer(env, online_learner=online_learner, target_fps=args.fps)
    viewer.run()


if __name__ == "__main__":
    main()
