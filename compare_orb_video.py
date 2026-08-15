import cv2
import csv
import argparse
import numpy as np
from pathlib import Path


def resize_keep_ratio(img, max_width):
    if max_width <= 0 or img.shape[1] <= max_width:
        return img

    scale = max_width / img.shape[1]
    return cv2.resize(
        img,
        (max_width, int(img.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def blur_score(gray):
    # 越大通常越清晰
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def analyze_video(
    video_path,
    output_csv,
    duration=10.0,
    sample_hz=30.0,
    max_width=1352,
    nfeatures=2500,
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Invalid video FPS")

    # 两个视频即使 FPS 不同，也按相同 sample_hz 比较
    stride = max(1, round(fps / sample_hz))
    max_frame = min(total_frames, int(duration * fps))

    print(f"\n===== {video_path} =====")
    print(f"fps          : {fps:.3f}")
    print(f"frames       : {total_frames}")
    print(f"duration test: {max_frame / fps:.2f} s")
    print(f"stride       : {stride}")
    print(f"effective Hz : {fps / stride:.2f}")

    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=20,
    )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    prev_gray = None
    prev_kp = None
    prev_des = None
    prev_frame_idx = None

    rows = []

    frame_idx = 0

    while frame_idx < max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        frame = resize_keep_ratio(frame, max_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp, des = orb.detectAndCompute(gray, None)

        current_blur = blur_score(gray)
        current_brightness = float(gray.mean())

        if (
            prev_gray is not None
            and prev_des is not None
            and des is not None
            and len(prev_des) >= 2
            and len(des) >= 2
        ):
            # KNN + Lowe ratio test
            knn = matcher.knnMatch(prev_des, des, k=2)

            good = []

            for pair in knn:
                if len(pair) < 2:
                    continue

                m, n = pair

                if m.distance < 0.75 * n.distance:
                    good.append(m)

            inliers = 0
            inlier_ratio = 0.0
            mean_motion = np.nan

            if len(good) >= 8:
                pts1 = np.float32(
                    [prev_kp[m.queryIdx].pt for m in good]
                )

                pts2 = np.float32(
                    [kp[m.trainIdx].pt for m in good]
                )

                # 不依赖相机内参，先用 Fundamental matrix 做诊断
                F, mask = cv2.findFundamentalMat(
                    pts1,
                    pts2,
                    cv2.FM_RANSAC,
                    1.5,
                    0.99,
                )

                if mask is not None:
                    mask = mask.ravel().astype(bool)

                    inliers = int(mask.sum())
                    inlier_ratio = inliers / len(good)

                    if inliers > 0:
                        displacement = np.linalg.norm(
                            pts2[mask] - pts1[mask],
                            axis=1,
                        )
                        mean_motion = float(np.mean(displacement))

            row = {
                "frame0": prev_frame_idx,
                "frame1": frame_idx,
                "time_s": frame_idx / fps,
                "kp0": len(prev_kp),
                "kp1": len(kp),
                "matches": len(good),
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "mean_motion_px": mean_motion,
                "blur": current_blur,
                "brightness": current_brightness,
            }

            rows.append(row)

        prev_gray = gray
        prev_kp = kp
        prev_des = des
        prev_frame_idx = frame_idx

        frame_idx += 1

    cap.release()

    if not rows:
        raise RuntimeError("No frame pairs analyzed")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def arr(name):
        return np.asarray(
            [x[name] for x in rows],
            dtype=float,
        )

    def stats(name):
        x = arr(name)
        x = x[np.isfinite(x)]

        return (
            np.mean(x),
            np.median(x),
            np.percentile(x, 10),
            np.percentile(x, 90),
        )

    print("\n----- SUMMARY -----")

    for name in [
        "kp1",
        "matches",
        "inliers",
        "inlier_ratio",
        "mean_motion_px",
        "blur",
        "brightness",
    ]:
        mean, median, p10, p90 = stats(name)

        print(
            f"{name:16s}"
            f" mean={mean:9.3f}"
            f" median={median:9.3f}"
            f" p10={p10:9.3f}"
            f" p90={p90:9.3f}"
        )

    low_inliers = np.mean(arr("inliers") < 50)
    very_low_inliers = np.mean(arr("inliers") < 15)

    print()
    print(f"pairs with inliers < 50 : {100*low_inliers:.1f}%")
    print(f"pairs with inliers < 15 : {100*very_low_inliers:.1f}%")
    print(f"CSV saved to: {output_csv}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=30.0)
    parser.add_argument("--max-width", type=int, default=1352)
    parser.add_argument("--nfeatures", type=int, default=2500)

    args = parser.parse_args()

    analyze_video(
        args.video,
        args.output,
        duration=args.duration,
        sample_hz=args.sample_hz,
        max_width=args.max_width,
        nfeatures=args.nfeatures,
    )