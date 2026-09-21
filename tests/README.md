# Dataset checks

Run from the project root with the project's Python environment.

```bash
PYTHONPATH=. python tests/check_episode_motion.py
PYTHONPATH=. python tests/check_negative_z.py
python tests/check_dataset_plan.py
python tests/check_dataset_plan.py /path/to/dataset_plan.pkl
```

The Zarr scripts use the dataset paths defined inside each script and require
the project's Zarr codecs. The plan checker requires NumPy and SciPy and defaults
to `/home/hjadmin/Downloads/demo3/dataset_plan.pkl`. Only load trusted pickle files.

The plan checker reports position bounds, gripper width bounds, negative Z counts,
linear and angular speed quantiles, and position jumps of at least 0.10 m for each
episode and gripper. Speeds use episode timestamps in seconds. Frame indices are
zero-based; global indices refer to concatenated plan episodes, not source video
frames. Invalid time intervals are excluded from speeds and reported as warnings.
