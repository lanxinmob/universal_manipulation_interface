"""Report per-episode height statistics and negative Z coordinates."""

import numpy as np
import zarr

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


def main():
    register_codecs()
    path = "/home/hjadmin/Downloads/demo4/dataset.zarr.zip"

    with zarr.ZipStore(path, mode="r") as store:
        root = zarr.group(store=store)
        z = root["data"]["robot0_eef_pos"][:, 2]
        ends = root["meta"]["episode_ends"][:]

        start = 0
        for ep, end in enumerate(ends):
            end = int(end)
            ze = z[start:end]
            neg = ze < 0
            print(
                f"episode {ep}:",
                f"min={ze.min():.4f} m,",
                f"median={np.median(ze):.4f} m,",
                f"negative={neg.sum()}/{len(ze)}",
                f"({neg.mean()*100:.1f}%)",
                f"argmin_local={np.argmin(ze)}",
            )
            start = end


if __name__ == "__main__":
    main()
