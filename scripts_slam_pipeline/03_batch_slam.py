"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/fold_cloth_20231214/demos
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import csv
import math
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import cv2
import av
import numpy as np
from umi.common.cv_util import draw_predefined_mask


# %%
TRAJECTORY_COLUMNS = {
    'frame_idx', 'timestamp', 'is_lost',
    'x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w'
}
MIN_TRACKED_FRAMES = 60
MAX_LOST_FRAMES = 10


def validate_trajectory(csv_path):
    """Return None when a trajectory is usable by the dataset planner."""
    if not csv_path.is_file():
        return 'camera_trajectory.csv was not created'

    try:
        with csv_path.open(newline='') as file:
            reader = csv.DictReader(file)
            columns = set(reader.fieldnames or ())
            missing_columns = sorted(TRAJECTORY_COLUMNS - columns)
            if missing_columns:
                return f"missing columns: {', '.join(missing_columns)}"

            previous_timestamp = None
            tracked_frames = 0
            lost_frames = 0
            row_count = 0
            for row_count, row in enumerate(reader, start=1):
                try:
                    timestamp = float(row['timestamp'])
                    pose = [float(row[name]) for name in (
                        'x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w')]
                except (TypeError, ValueError):
                    return f'invalid numeric value at data row {row_count}'

                if not math.isfinite(timestamp) or not all(
                        math.isfinite(value) for value in pose):
                    return f'non-finite value at data row {row_count}'
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    return f'timestamps are not strictly increasing at data row {row_count}'
                previous_timestamp = timestamp

                is_lost = row['is_lost'].strip().lower()
                if is_lost == 'true':
                    lost_frames += 1
                elif is_lost == 'false':
                    tracked_frames += 1
                else:
                    return f"invalid is_lost value at data row {row_count}: {row['is_lost']!r}"
    except (OSError, csv.Error) as error:
        return f'could not read trajectory: {error}'

    if row_count == 0:
        return 'trajectory has no data rows'
    if lost_frames > MAX_LOST_FRAMES:
        return f'too many lost frames: {lost_frames} > {MAX_LOST_FRAMES}'
    if tracked_frames < MIN_TRACKED_FRAMES:
        return f'too few tracked frames: {tracked_frames} < {MIN_TRACKED_FRAMES}'
    return None


def runner(cmd, cwd, stdout_path, stderr_path, timeout, **kwargs):
    try:
        return subprocess.run(cmd,                       
            cwd=str(cwd),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w'),
            timeout=timeout,
            **kwargs)
    except subprocess.TimeoutExpired as e:
        return e


def collect_results(completed, future_video_dirs, failures, pbar):
    for future in completed:
        video_dir = future_video_dirs[future]
        try:
            result = future.result()
        except Exception as error:
            failures.append((video_dir, f'runner failed: {error}'))
            continue

        if isinstance(result, subprocess.TimeoutExpired):
            failures.append((video_dir, f'timed out after {result.timeout:.1f}s'))
            continue
        if result.returncode != 0:
            failures.append((video_dir, f'docker exited with code {result.returncode}'))
            continue

        validation_error = validate_trajectory(video_dir.joinpath('camera_trajectory.csv'))
        if validation_error is not None:
            failures.append((video_dir, validation_error))
    pbar.update(len(completed))


# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="orb_slam3:gopro13")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=36, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
def main(input_dir, map_path, docker_image, num_workers, max_lost_frames, timeout_multiple, no_docker_pull):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')
    
    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file()

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 4)

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    failures = list()
    with tqdm(total=len(input_video_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            future_video_dirs = dict()
            for video_dir in tqdm(input_video_dirs):
                video_dir = video_dir.absolute()
                trajectory_path = video_dir.joinpath('camera_trajectory.csv')
                validation_error = validate_trajectory(trajectory_path)
                if validation_error is None:
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    pbar.update(1)
                    continue
                if trajectory_path.is_file():
                    print(f"Invalid existing trajectory for {video_dir.name}: {validation_error}; rerunning")
                
                # softlink won't work in bind volume
                mount_target = pathlib.Path('/data')
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                video_path = mount_target.joinpath('raw_video.mp4')
                json_path = mount_target.joinpath('imu_data.json')
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')
                
                # find video duration
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple
                
                slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
                slam_mask = draw_predefined_mask(
                    slam_mask, color=255, mirror=True, gripper=False, finger=True)
                cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                map_mount_source = map_path
                map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

                # GoPro13 calibration
                calib_mount_source = pathlib.Path(ROOT_DIR).joinpath('example','calibration_gopro13').absolute()
                calib_mount_target = pathlib.Path('/calibration')

                setting_path = calib_mount_source.joinpath('gopro13_2_7k_wide_fisheye.yaml')
                assert setting_path.is_file(), f"Missing SLAM setting: {setting_path}"

                # run SLAM
                cmd = [
                    'docker',
                    'run',
                    '--rm', # delete after finish
                    '--volume', str(video_dir) + ':' + '/data',
                    '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
                    '--volume',str(calib_mount_source) + ':' + str(calib_mount_target) + ':ro',
                    docker_image,
                    '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
                    '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
                    '--setting', '/calibration/gopro13_2_7k_wide_fisheye.yaml',
                    '--input_video', str(video_path),
                    '--input_imu_json', str(json_path),
                    '--output_trajectory_csv', str(csv_path),
                    '--load_map', str(map_mount_target),
                    '--mask_img', str(mask_path),
                    '--max_lost_frames', str(max_lost_frames)
                ]

                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    collect_results(completed, future_video_dirs, failures, pbar)

                future = executor.submit(
                    runner, cmd, str(video_dir), stdout_path, stderr_path, timeout)
                futures.add(future)
                future_video_dirs[future] = video_dir
                # print(' '.join(cmd))

            completed, futures = concurrent.futures.wait(futures)
            collect_results(completed, future_video_dirs, failures, pbar)

    if failures:
        click.echo(f'\nSLAM failed for {len(failures)} video(s):', err=True)
        for video_dir, reason in failures:
            click.echo(
                f'  {video_dir.name}: {reason} '
                f'(logs: {video_dir.joinpath("slam_stdout.txt")}, '
                f'{video_dir.joinpath("slam_stderr.txt")})',
                err=True)
        raise click.ClickException(
            'Batch SLAM did not produce a valid trajectory for every video; '
            'stopping the pipeline.')

    print('Done! All trajectories are valid.')

# %%
if __name__ == "__main__":
    main()
