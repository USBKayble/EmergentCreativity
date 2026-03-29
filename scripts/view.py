"""
scripts/view.py
===============
Console entry-point for the interactive viewer.

Usage::

    python -m scripts.view                      # manual control
    python -m scripts.view --nn checkpoints/checkpoint_100000.pt
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
    parser.add_argument("--nn",     type=str,  default=None, help="NN checkpoint to load")
    parser.add_argument("--gui",    action="store_true",     help="Open PyBullet GUI")
    parser.add_argument("--fps",    type=int,  default=30)
    parser.add_argument("--config", type=str,  default=None)
    args = parser.parse_args()

    from src.emergent_creativity.sim_env import TenantEnv
    from src.emergent_creativity.ui.viewer import SimViewer

    nn_agent = None
    if args.nn:
        try:
            import torch
            import numpy as np
            from src.emergent_creativity.nn.architecture import TenantNetwork
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = TenantNetwork()
            ck  = torch.load(args.nn, map_location=device)
            net.load_state_dict(ck["model"])
            net.to(device)
            net.eval()
            lstm_state = [None]

            def nn_agent(obs, _):
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

        except Exception as e:
            print(f"[view] Could not load NN: {e}")

    env = TenantEnv(gui=args.gui, config_path=args.config)
    viewer = SimViewer(env, nn_agent=nn_agent, target_fps=args.fps)
    viewer.run()


if __name__ == "__main__":
    main()
