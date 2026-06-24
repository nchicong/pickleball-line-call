import cv2
import numpy as np
import json
import os
import argparse
from ultralytics import YOLO


BALL_CLASS_ID = 0
TRAIL_LEN = 20
COURT_LENGTH = 44.0
COURT_WIDTH = 20.0


def load_calibration(path):
    with open(path) as f:
        calib = json.load(f)
    M = np.array(calib["M"], dtype=np.float32)
    Minv = np.array(calib["Minv"], dtype=np.float32)
    return M, Minv, calib


def load_analysis(path):
    with open(path) as f:
        return json.load(f)


def draw_court_overlay(frame, Minv):
    court_pts = [
        (0, 0),
        (COURT_WIDTH, 0),
        (COURT_WIDTH, COURT_LENGTH),
        (0, COURT_LENGTH),
    ]
    src_pts = np.array(court_pts, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(src_pts, Minv).astype(np.int32)

    cv2.polylines(frame, [img_pts], True, (0, 255, 0), 2)

    kitchen_y = 7.0
    kitchen_pts = [
        (0, kitchen_y),
        (COURT_WIDTH, kitchen_y),
        (COURT_WIDTH, COURT_LENGTH - kitchen_y),
        (0, COURT_LENGTH - kitchen_y),
    ]
    src_pts = np.array(kitchen_pts, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(src_pts, Minv).astype(np.int32)
    cv2.polylines(frame, [img_pts[:2]], True, (0, 255, 255), 2)
    cv2.polylines(frame, [img_pts[2:]], True, (0, 255, 255), 2)

    center_pts = [(COURT_WIDTH / 2, 0), (COURT_WIDTH / 2, COURT_LENGTH)]
    src_pts = np.array(center_pts, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(src_pts, Minv).astype(np.int32)
    cv2.line(frame, tuple(img_pts[0, 0]), tuple(img_pts[1, 0]), (255, 255, 0), 2)

    baseline_pts = [(0, 0), (COURT_WIDTH, 0)]
    src_pts = np.array(baseline_pts, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(src_pts, Minv).astype(np.int32)
    cv2.line(frame, tuple(img_pts[0, 0]), tuple(img_pts[1, 0]), (0, 0, 255), 2)

    return frame


def draw_ball(frame, detections, Minv, trail):
    for d in detections:
        bbox = d["bbox"]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cx, cy = int(d["image_cx"]), int(d["image_cy"])
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        conf = d.get("confidence", 0)
        cv2.putText(frame, f"{conf:.2f}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for pt in trail:
        court_pt = np.array([[pt["x"], pt["y"]]], dtype=np.float32).reshape(-1, 1, 2)
        img_pt = cv2.perspectiveTransform(court_pt, Minv).astype(np.int32)
        cv2.circle(frame, tuple(img_pt[0, 0]), 3, (0, 255, 255), -1)

    return frame


def draw_info(frame, frame_idx, total_frames, in_rally, out_calls, rally_count):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    state = "RALLY" if in_rally else "IDLE"
    color = (0, 255, 0) if in_rally else (128, 128, 128)
    cv2.putText(frame, f"State: {state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    cv2.putText(frame, f"Out calls: {len(out_calls)}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, f"Rallies: {rally_count}", (200, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--analysis", default=None)
    parser.add_argument("--output", default="data/results/visualized.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    video_path = args.video if os.path.isabs(args.video) else os.path.join(base, args.video)
    calib_path = args.calibration if os.path.isabs(args.calibration) else os.path.join(base, args.calibration)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(base, args.output)
    analysis_path = args.analysis if args.analysis and (os.path.isabs(args.analysis) or os.path.exists(args.analysis)) else None
    if analysis_path is None and args.analysis:
        analysis_path = os.path.join(base, args.analysis)

    M, Minv, _ = load_calibration(calib_path)

    analysis = None
    if analysis_path:
        analysis = load_analysis(analysis_path)
        print(f"[INFO] Loaded analysis: {analysis_path}")
        out_calls_list = analysis.get("out_calls", [])
        rallies_list = analysis.get("rallies", [])
    else:
        out_calls_list = []
        rallies_list = []

    model = YOLO(args.model)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = min(args.start_frame, video_total - 1)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = video_total
    if args.max_frames > 0:
        end_frame = min(video_total, start_frame + args.max_frames)

    total_frames = end_frame

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    out_call_frames = {c["frame"]: c for c in out_calls_list}
    rally_frames = set()
    for r in rallies_list:
        for f in range(r["start_frame"], r["end_frame"] + 1):
            rally_frames.add(f)

    trail = []
    frame_idx = start_frame

    print(f"[INFO] Rendering frames {start_frame}-{end_frame - 1} to {output_path}")

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != BALL_CLASS_ID or conf < args.conf:
                    continue
                xyxy = box.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0
                detections.append(
                    {
                        "image_cx": cx,
                        "image_cy": cy,
                        "confidence": conf,
                        "bbox": xyxy,
                    }
                )

        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            court_pt = np.array([[best["image_cx"], best["image_cy"]]], dtype=np.float32).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(court_pt, M)
            trail.append({"x": warped[0, 0, 0], "y": warped[0, 0, 1]})
            if len(trail) > TRAIL_LEN:
                trail.pop(0)
        else:
            if trail:
                trail.pop(0)

        in_rally = frame_idx in rally_frames

        frame = draw_court_overlay(frame, Minv)
        frame = draw_ball(frame, detections, Minv, trail)
        frame = draw_info(frame, frame_idx, total_frames, in_rally, out_calls_list, len(rallies_list))

        if frame_idx in out_call_frames:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            cv2.putText(frame, "OUT!", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 500 == 0:
            print(f"[RENDER] Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    print(f"[DONE] Video saved to {output_path}")


if __name__ == "__main__":
    main()
