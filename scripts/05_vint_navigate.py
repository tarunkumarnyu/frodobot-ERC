#!/usr/bin/env python3
"""Navigate using ViNT — either to a goal image or along a topomap.

Usage:
    # Navigate to a specific image:
    python scripts/05_vint_navigate.py --goal target.jpg

    # Follow a recorded topomap:
    python scripts/05_vint_navigate.py --topomap data/topomap

    # Follow topomap to a specific node:
    python scripts/05_vint_navigate.py --topomap data/topomap --goal-node 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goal", help="Goal image path (navigate directly to this image)")
    p.add_argument("--topomap", help="Topomap directory (follow recorded path)")
    p.add_argument("--goal-node", type=int, default=-1, help="Goal node in topomap (-1 = last)")
    p.add_argument("--weights", default=str(Path.home() / "visualnav-transformer/deployment/model_weights/checkpoints/vint.pth"))
    p.add_argument("--config", default=str(Path.home() / "visualnav-transformer/train/config/vint.yaml"))
    p.add_argument("--sdk-url", default="http://localhost:8000")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--max-v", type=float, default=0.3)
    p.add_argument("--max-w", type=float, default=0.4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if not args.goal and not args.topomap:
        p.error("Provide --goal (image path) or --topomap (directory)")

    from topnav.vint_navigator import ViNTNavigator

    print("Loading ViNT...")
    nav = ViNTNavigator(
        weights_path=args.weights,
        config_path=args.config,
        sdk_url=args.sdk_url,
        device=args.device,
        max_v=args.max_v,
        max_w=args.max_w,
    )
    print("ViNT ready!")

    if args.topomap:
        success = nav.navigate_topomap(
            args.topomap, goal_node=args.goal_node, max_steps=args.max_steps)
    else:
        success = nav.navigate_to_image(args.goal, max_steps=args.max_steps)

    print(f"\nNavigation {'SUCCEEDED' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
