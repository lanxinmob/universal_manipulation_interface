"""Report episode motion statistics and position jumps in a Zarr dataset."""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


def main():
    register_codecs()
    path = "/home/hjadmin/Downloads/demo_towel/towel_0.zarr.zip"
    stored_hz = 29.97

    with zarr.ZipStore(path, mode="r") as store:
        root = zarr.group(store=store)
        data = root["data"]
        ends = root["meta"]["episode_ends"][:]
        print("episode_ends:", ends)

        start = 0
        for ep, end in enumerate(ends):
            end = int(end)
            pos = data["robot0_eef_pos"][start:end]
            rot = data["robot0_eef_rot_axis_angle"][start:end]
            width = data["robot0_gripper_width"][start:end, 0]

            step_distance = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            linear_speed = step_distance * stored_hz
            rotations = Rotation.from_rotvec(rot)
            relative_rotations = rotations[:-1].inv() * rotations[1:]
            angular_speed = relative_rotations.magnitude() * stored_hz

            print(f"\nEPISODE {ep}: frames={end-start}")
            print("xyz min:", pos.min(axis=0))
            print("xyz max:", pos.max(axis=0))
            print("width range:", width.min(), width.max())
            print("linear speed m/s p50,p95,max:",
                  np.quantile(linear_speed, [0.5, 0.95, 1.0]))
            print("angular speed rad/s p50,p95,max:",
                  np.quantile(angular_speed, [0.5, 0.95, 1.0]))

            for i in np.flatnonzero(step_distance >= 0.10):
                print(
                    f"WARNING: EPISODE {ep} "
                    f"local frames {i} -> {i+1} "
                    f"(global frames {start+i} -> {start+i+1}) "
                    f"position jump {step_distance[i]:.4f} m"
                )

            start = end


if __name__ == "__main__":
    main()
