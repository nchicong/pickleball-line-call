import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path


COURT_WIDTH = 20.0
COURT_LENGTH = 44.0
KITCHEN_DEPTH = 7.0

CORNER_LABELS = [
    "Far-Left (xa nhất, bên trái)",
    "Far-Right (xa nhất, bên phải)",
    "Near-Left (gần nhất, bên trái)",
    "Near-Right (gần nhất, bên phải)",
]

points = []
current_label_idx = 0
img_display = None


def mouse_callback(event, x, y, flags, param):
    global points, current_label_idx, img_display
    if event == cv2.EVENT_LBUTTONDOWN and current_label_idx < 4:
        points.append((x, y))
        cv2.circle(img_display, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            img_display,
            f"{current_label_idx + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        current_label_idx += 1
        if current_label_idx < 4:
            update_instruction()
        cv2.imshow("Calibrate Court", img_display)


def update_instruction():
    global img_display
    overlay = img_display.copy()
    cv2.rectangle(overlay, (5, 5), (600, 35), (0, 0, 0), -1)
    img_display = cv2.addWeighted(img_display, 0, overlay, 1, 0)
    label = f"Click: {CORNER_LABELS[current_label_idx]}"
    cv2.putText(img_display, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    global points, current_label_idx, img_display

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="data/calibration/court_calib.json", help="Output calibration path")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to use for calibration")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    video_path = args.video if os.path.isabs(args.video) else os.path.join(base, args.video)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(base, args.output)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = min(args.frame_idx, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"[ERROR] Cannot read frame {target_frame} from {video_path}")
        return

    img_display = frame.copy()
    update_instruction()
    cv2.imshow("Calibrate Court", img_display)
    cv2.setMouseCallback("Calibrate Court", mouse_callback)

    print("[INFO] Click the 4 court corners in order. Press 'r' to reset. Press 'q' to quit without saving.")
    while current_label_idx < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            points = []
            current_label_idx = 0
            img_display = frame.copy()
            update_instruction()
            cv2.imshow("Calibrate Court", img_display)
            print("[INFO] Reset. Click corners again.")
        elif key == ord("q"):
            print("[INFO] Calibration cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    src_pts = np.array(points, dtype=np.float32)

    dst_pts = np.array(
        [
            [0, 0],
            [COURT_WIDTH, 0],
            [0, COURT_LENGTH],
            [COURT_WIDTH, COURT_LENGTH],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    calib = {
        "image_shape": list(frame.shape),
        "frame_index": target_frame,
        "source_points": points,
        "destination_points": dst_pts.tolist(),
        "M": M.tolist(),
        "Minv": Minv.tolist(),
        "court_width_ft": COURT_WIDTH,
        "court_length_ft": COURT_LENGTH,
        "kitchen_depth_ft": KITCHEN_DEPTH,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"[OK] Calibration saved to {output_path}")
    print(f"[OK] Source points: {points}")
    print(f"[OK] Transform matrix: {M}")


if __name__ == "__main__":
    main()
