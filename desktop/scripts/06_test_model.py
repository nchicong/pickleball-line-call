import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt or .tflite model")
    parser.add_argument("--source", required=True, help="Path to image, video, or frames directory")
    parser.add_argument("--output_dir", default="data/test_results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), "..")
    output_dir = os.path.join(base, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    source = args.source
    if not os.path.isabs(source):
        source = os.path.join(base, source)

    print(f"[INFO] Running inference on: {source}")
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        save_txt=True,
        project=output_dir,
        name="detect",
        verbose=True,
    )

    print(f"[DONE] Results saved to: {os.path.join(output_dir, 'detect')}")
    ball_count = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"[INFO] Total balls detected across all frames: {ball_count}")


if __name__ == "__main__":
    main()
