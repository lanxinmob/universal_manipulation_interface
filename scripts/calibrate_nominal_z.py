import collections
import json
import math
import pathlib
import pickle
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import click


TAG_PER_GRIPPER = 6


def identify_gripper(
        frames: Sequence[Mapping[str, Any]],
        tag_det_threshold: float) -> Tuple[int, float]:
    if not frames:
        raise ValueError("Tag detection input is empty.")

    tag_counts: Dict[int, int] = collections.defaultdict(int)
    for frame in frames:
        for tag_id in frame.get('tag_dict', {}):
            tag_counts[int(tag_id)] += 1

    if not tag_counts:
        raise ValueError("No tags were detected.")

    n_frames = len(frames)
    max_gripper_id = max(tag_counts) // TAG_PER_GRIPPER
    gripper_probabilities: Dict[int, float] = {}
    for gripper_id in range(max_gripper_id + 1):
        left_id = gripper_id * TAG_PER_GRIPPER
        right_id = left_id + 1
        probability = min(
            tag_counts[left_id] / n_frames,
            tag_counts[right_id] / n_frames)
        if probability > 0:
            gripper_probabilities[gripper_id] = probability

    if not gripper_probabilities:
        raise ValueError("No gripper finger tag pair was detected.")

    gripper_id = max(gripper_probabilities, key=gripper_probabilities.get)
    probability = gripper_probabilities[gripper_id]
    if probability < tag_det_threshold:
        raise ValueError(
            f"Gripper tag detection rate {probability:.3f} is below "
            f"the required threshold {tag_det_threshold:.3f}.")
    return gripper_id, probability


def collect_tag_depths(
        frames: Sequence[Mapping[str, Any]], tag_id: int) -> List[float]:
    depths: List[float] = []
    for frame in frames:
        tag = frame.get('tag_dict', {}).get(tag_id)
        if tag is None:
            continue
        tvec = tag.get('tvec', [])
        if hasattr(tvec, 'reshape'):
            tvec = tvec.reshape(-1)
        if len(tvec) == 0:
            continue
        z = float(tvec[-1])
        if math.isfinite(z) and z > 0:
            depths.append(z)
    return depths


def linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - weight) +
        sorted_values[upper_index] * weight)


def estimate_nominal_z(
        frames: Sequence[Mapping[str, Any]],
        gripper_id: int,
        quantile: float,
        margin: float) -> Dict[str, Any]:
    if not 0 <= quantile < 0.5:
        raise ValueError("quantile must be in the interval [0, 0.5).")
    if margin < 0:
        raise ValueError("margin must be non-negative.")

    tag_ids = (
        gripper_id * TAG_PER_GRIPPER,
        gripper_id * TAG_PER_GRIPPER + 1)
    tag_statistics: Dict[str, Dict[str, float]] = {}
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    for tag_id in tag_ids:
        depths = collect_tag_depths(frames, tag_id)
        if len(depths) < 10:
            raise ValueError(
                f"Tag {tag_id} has only {len(depths)} valid depth samples; "
                "at least 10 are required.")

        depths.sort()
        lower = linear_quantile(depths, quantile)
        upper = linear_quantile(depths, 1.0 - quantile)
        median = float(statistics.median(depths))
        mad = float(statistics.median(abs(z - median) for z in depths))
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        tag_statistics[str(tag_id)] = {
            'sample_count': len(depths),
            'median_z': median,
            'mad_z': mad,
            'lower_quantile_z': lower,
            'upper_quantile_z': upper,
        }

    z_min = min(lower_bounds) - margin
    z_max = max(upper_bounds) + margin
    return {
        'gripper_id': gripper_id,
        'left_finger_tag_id': tag_ids[0],
        'right_finger_tag_id': tag_ids[1],
        'quantile': quantile,
        'margin': margin,
        'nominal_z': (z_min + z_max) / 2.0,
        'z_tolerance': (z_max - z_min) / 2.0,
        'z_min': z_min,
        'z_max': z_max,
        'tag_statistics': tag_statistics,
    }


@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
              help='Tag detection pkl from a gripper calibration recording.')
@click.option('-o', '--output', 'output_path', default=None,
              type=click.Path(dir_okay=False, path_type=pathlib.Path),
              help='Output JSON. Defaults to gripper_nominal_z.json beside the input.')
@click.option('-t', '--tag-det-threshold', type=click.FloatRange(0.0, 1.0),
              default=0.8, show_default=True)
@click.option('-q', '--quantile', type=click.FloatRange(0.0, 0.499999),
              default=0.01, show_default=True,
              help='Tail fraction excluded on each side before adding margin.')
@click.option('-m', '--margin', type=click.FloatRange(min=0.0),
              default=0.001, show_default=True,
              help='Extra depth margin in meters on both sides.')
def main(input_path: pathlib.Path, output_path: Optional[pathlib.Path],
         tag_det_threshold: float, quantile: float, margin: float) -> None:
    if output_path is None:
        output_path = input_path.with_name('gripper_nominal_z.json')

    with input_path.open('rb') as input_file:
        frames = pickle.load(input_file)

    gripper_id, detection_probability = identify_gripper(
        frames, tag_det_threshold)
    result = estimate_nominal_z(frames, gripper_id, quantile, margin)
    result['detection_probability'] = detection_probability

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write('\n')

    print(
        f"Detected gripper id {gripper_id} with probability "
        f"{detection_probability:.3f}")
    for tag_id, statistics in result['tag_statistics'].items():
        print(
            f"Tag {tag_id}: n={statistics['sample_count']}, "
            f"median={statistics['median_z']:.6f} m, "
            f"range=[{statistics['lower_quantile_z']:.6f}, "
            f"{statistics['upper_quantile_z']:.6f}] m")
    print(f"nominal_z={result['nominal_z']:.6f} m")
    print(f"z_tolerance={result['z_tolerance']:.6f} m")
    print(f"accepted range=[{result['z_min']:.6f}, {result['z_max']:.6f}] m")
    print(f"Saved calibration to {output_path}")


if __name__ == '__main__':
    main()
