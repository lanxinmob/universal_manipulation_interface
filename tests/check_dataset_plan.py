"""Inspect motion and height statistics in a trusted dataset_plan.pkl."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path,
        default=Path("/home/hjadmin/Downloads/demo3/dataset_plan.pkl"),
        help="Dataset plan path (only load trusted pickle files)",
    )
    args = parser.parse_args()
    with args.path.expanduser().open("rb") as stream:
        plan = pickle.load(stream)

    print(f"dataset: {args.path}")
    print(f"episodes: {len(plan)}")
    start = 0
    for ep, episode in enumerate(plan):
        timestamps = np.asarray(episode["episode_timestamps"], dtype=float)
        if timestamps.ndim != 1:
            raise ValueError(f"Episode {ep}: timestamps must be one-dimensional")
        frames = len(timestamps)
        dt = np.diff(timestamps)
        valid_dt = np.isfinite(dt) & (dt > 0)
        if not valid_dt.all():
            print(f"WARNING: EPISODE {ep}: invalid time intervals at "
                  f"local frames {np.flatnonzero(~valid_dt).tolist()}; "
                  "excluded from speed statistics")

        for robot, gripper in enumerate(episode["grippers"]):
            label = f"EPISODE {ep} robot{robot}"
            print(f"\n{label}: frames={frames}")
            pos_rot = np.asarray(gripper["tcp_pose"], dtype=float)
            width = np.asarray(gripper["gripper_width"], dtype=float).reshape(-1)
            if pos_rot.shape != (frames, 6) or width.shape != (frames,):
                raise ValueError(f"{label}: pose/width shapes do not match timestamps")
            if frames == 0:
                print("WARNING: empty episode")
                continue
            if not np.isfinite(pos_rot).all() or not np.isfinite(width).all():
                print("WARNING: non-finite pose or width values; skipping statistics")
                continue

            pos = pos_rot[:, :3]
            z = pos[:, 2]
            print("xyz min:", pos.min(axis=0))
            print("xyz max:", pos.max(axis=0))
            print("width range:", width.min(), width.max())
            print(f"z: min={z.min():.4f} m, median={np.median(z):.4f} m, "
                  f"negative={(z < 0).sum()}/{frames} "
                  f"({(z < 0).mean()*100:.1f}%), argmin_local={np.argmin(z)}")

            step_distance = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            if valid_dt.any():
                rotations = Rotation.from_rotvec(pos_rot[:, 3:])
                angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
                print("linear speed m/s p50,p95,max:",
                      np.quantile(step_distance[valid_dt] / dt[valid_dt], [0.5, 0.95, 1.0]))
                print("angular speed rad/s p50,p95,max:",
                      np.quantile(angles[valid_dt] / dt[valid_dt], [0.5, 0.95, 1.0]))
            else:
                print("Speed statistics unavailable: no valid time intervals")

            for i in np.flatnonzero(step_distance >= 0.10):
                print(f"WARNING: {label} local frames {i} -> {i+1} "
                      f"(global frames {start+i} -> {start+i+1}) "
                      f"position jump {step_distance[i]:.4f} m")
        start += frames


if __name__ == "__main__":
    main()
