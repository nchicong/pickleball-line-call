import cv2
import numpy as np
import json
import os
import argparse
from ultralytics import YOLO
from collections import deque


TRAJECTORY_LEN = 15
BOUNCE_WINDOW = 5
CONF_THRESH = 0.25
DEBOUNCE_FRAMES = 3
RALLY_TIMEOUT_FRAMES = 150
SERVE_Y_THRESH = 0.25
SERVE_MIN_FRAMES = 5
KITCHEN_Y = 7.0
COURT_LENGTH = 44.0
COURT_WIDTH = 20.0

BALL_CLASS_ID = 0


def load_calibration(path):
    with open(path) as f:
        calib = json.load(f)
    M = np.array(calib["M"], dtype=np.float32)
    Minv = np.array(calib["Minv"], dtype=np.float32)
    return M, Minv, calib


def warp_point(pt, M):
    pt_h = np.array([[pt[0], pt[1]]], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pt_h, M)
    return (warped[0, 0, 0], warped[0, 0, 1])


def is_ball_in_bounds(x, y):
    return 0 <= x <= COURT_WIDTH and 0 <= y <= COURT_LENGTH


def detect_serve(trajectory):
    if len(trajectory) < SERVE_MIN_FRAMES:
        return False
    first_y = trajectory[0][1]
    if first_y > COURT_LENGTH * SERVE_Y_THRESH:
        return False
    y_vals = [p[1] for p in trajectory]
    dy = y_vals[-1] - y_vals[0]
    if dy < 0:
        return False
    kitchen_crossings = sum(
        1 for p in trajectory if p[1] > KITCHEN_Y
    )
    if kitchen_crossings < 3:
        return False
    return True


def bouncing_trajectory(trajectory):
    if len(trajectory) < 3:
        return False
    speeds = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        speeds.append(np.sqrt(dx * dx + dy * dy))
    avg_speed = np.mean(speeds)
    if avg_speed < 0.5:
        return False
    return True


def detect_bounce(trajectory):
    if len(trajectory) < BOUNCE_WINDOW + 2:
        return -1, None
    y_vals = [p[1] for p in trajectory]
    window = BOUNCE_WINDOW
    for i in range(window, len(trajectory) - window):
        before = y_vals[i - window : i]
        after = y_vals[i : i + window + 1]
        dy_before = np.mean(np.diff(before)) if len(before) > 1 else 0
        dy_after = np.mean(np.diff(after)) if len(after) > 1 else 0
        if dy_before > 1.0 and dy_after < -1.0:
            return i, trajectory[i]
    return -1, None


def judge_out(bounce_pt):
    x, y = bounce_pt
    margin = 0.05
    if x < -margin or x > COURT_WIDTH + margin:
        return True
    if y < -margin or y > COURT_LENGTH + margin:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", default="data/results/analysis.json")
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--exclude_regions", nargs="+", default=[], help="Exclude regions x1,y1,x2,y2 (repeatable)")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    video_path = args.video if os.path.isabs(args.video) else os.path.join(base, args.video)
    calib_path = args.calibration if os.path.isabs(args.calibration) else os.path.join(base, args.calibration)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(base, args.output)

    M, Minv, calib = load_calibration(calib_path)

    exclude_zones = []
    for r in args.exclude_regions:
        parts = [int(x) for x in r.replace(" ", "").split(",")]
        if len(parts) == 4:
            exclude_zones.append(parts)
    if exclude_zones:
        print(f"[INFO] Excluding regions: {exclude_zones}")

    model = YOLO(args.model)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = min(args.start_frame, video_total - 1)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = video_total
    if args.max_frames > 0:
        end_frame = min(video_total, start_frame + args.max_frames)
    total_frames = end_frame

    total_frames = end_frame

    trajectory = deque(maxlen=TRAJECTORY_LEN)
    frame_idx = start_frame
    out_calls = []
    rallies = []
    in_rally = False
    rally_start = -1
    out_debounce = 0
    consecutive_detections = deque(maxlen=5)

    print(f"[INFO] Video: {video_path} ({video_total} frames, {fps:.1f} fps)")
    print(f"[INFO] Processing frames {start_frame} to {end_frame - 1} ({end_frame - start_frame} frames)")
    print(f"[INFO] Calibration: {calib_path}")

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )

        ball_court_positions = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != BALL_CLASS_ID or conf < args.conf:
                    continue
                xyxy = box.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                in_excluded = False
                for (x1, y1, x2, y2) in exclude_zones:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        in_excluded = True
                        break
                if in_excluded:
                    continue
                cx_warped, cy_warped = warp_point((cx, cy), M)
                ball_court_positions.append(
                    {
                        "frame": frame_idx,
                        "image_cx": cx,
                        "image_cy": cy,
                        "court_x": cx_warped,
                        "court_y": cy_warped,
                        "confidence": conf,
                        "bbox": xyxy,
                    }
                )

        if ball_court_positions:
            best = max(ball_court_positions, key=lambda b: b["confidence"])
            trajectory.append((best["court_x"], best["court_y"]))
            consecutive_detections.append(1)
        else:
            consecutive_detections.append(0)

        if not in_rally:
            if detect_serve(list(trajectory)):
                in_rally = True
                rally_start = frame_idx
                trajectory.clear()
                print(f"[RALLY] Started at frame {frame_idx}")
        else:
            if len(trajectory) >= 3 and bouncing_trajectory(list(trajectory)):
                bounce_idx, bounce_pt = detect_bounce(list(trajectory))
                if bounce_pt is not None:
                    is_out = judge_out(bounce_pt)
                    out_debounce = out_debounce + 1 if is_out else 0
                    if out_debounce >= DEBOUNCE_FRAMES:
                        print(f"[OUT] Frame {frame_idx}: bounce at ({float(bounce_pt[0]):.1f}, {float(bounce_pt[1]):.1f})")
                        out_calls.append(
                            {
                                "frame": frame_idx,
                                "bounce_x": float(bounce_pt[0]),
                                "bounce_y": float(bounce_pt[1]),
                                "is_out": True,
                            }
                        )
                        rallies.append(
                            {
                                "start_frame": rally_start,
                                "end_frame": frame_idx,
                                "out_frame": frame_idx,
                                "bounce": [float(bounce_pt[0]), float(bounce_pt[1])],
                            }
                        )
                        in_rally = False
                        out_debounce = 0
                        trajectory.clear()
            missed = sum(consecutive_detections)
            if missed >= 5:
                if len([c for c in consecutive_detections if c == 0]) >= 5:
                    print(f"[RALLY END] Frame {frame_idx}: ball lost >5 frames")
                    rallies.append(
                        {"start_frame": rally_start, "end_frame": frame_idx, "out_frame": -1, "bounce": None}
                    )
                    in_rally = False
                    trajectory.clear()

        if frame_idx % 500 == 0:
            print(f"[PROGRESS] Frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()

    results_data = {
        "video": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "calibration": calib_path,
        "out_calls": out_calls,
        "rallies": rallies,
        "settings": {
            "conf_thresh": args.conf,
            "trajectory_len": TRAJECTORY_LEN,
            "debounce_frames": DEBOUNCE_FRAMES,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[DONE] Analysis saved to {output_path}")
    print(f"[STATS] Total frames: {total_frames}")
    print(f"[STATS] Rallies detected: {len(rallies)}")
    print(f"[STATS] Out calls: {len(out_calls)}")


if __name__ == "__main__":
    main()
