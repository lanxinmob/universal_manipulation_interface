import numpy as np


def get_median_dt(timestamps, name):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        raise ValueError(f"{name} needs at least two timestamps.")
    deltas = np.diff(timestamps)
    if not np.all(np.isfinite(timestamps)) or np.any(deltas <= 0):
        raise ValueError(f"{name} timestamps must be finite and strictly increasing.")
    return float(np.median(deltas))


def match_nearest_timestamps(reference_timestamps, query_timestamps,
        max_delta, name):
    """Map each query timestamp to one unique, ordered reference sample."""
    reference_timestamps = np.asarray(reference_timestamps, dtype=np.float64)
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    get_median_dt(reference_timestamps, f"{name} reference")
    if len(query_timestamps) == 0:
        return np.empty(0, dtype=np.int64)
    if not np.all(np.isfinite(query_timestamps)):
        raise ValueError(f"{name} query timestamps must be finite.")

    right = np.searchsorted(reference_timestamps, query_timestamps, side='left')
    right = np.clip(right, 0, len(reference_timestamps) - 1)
    left = np.clip(right - 1, 0, len(reference_timestamps) - 1)
    use_right = (
        np.abs(reference_timestamps[right] - query_timestamps)
        < np.abs(reference_timestamps[left] - query_timestamps)
    )
    indices = np.where(use_right, right, left).astype(np.int64)
    deltas = np.abs(reference_timestamps[indices] - query_timestamps)

    if np.any(np.diff(indices) <= 0):
        raise ValueError(f"{name} matched duplicate or non-monotonic samples.")
    if np.any(deltas > max_delta):
        raise ValueError(
            f"{name} timestamp mismatch: max={deltas.max():.6f}s, "
            f"allowed={max_delta:.6f}s.")
    return indices
