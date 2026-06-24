"""
Orchestrator: run the full pipeline step by step.
Usage:
    python run_all.py --api_key YOUR_ROBOFLOW_KEY [--steps 1-5]
    python run_all.py [--steps 1-5]              # reads ROBOFLOW_API_KEY from .env
"""
import argparse
import os
import subprocess
import sys
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")


def run_step(script_name: str, description: str, args: list = None):
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}\n")
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, check=True)
    print(f"\n[OK] {description} completed.\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Pickleball training pipeline")
    parser.add_argument("--api_key", help="Roboflow API key (or set ROBOFLOW_API_KEY in .env)")
    parser.add_argument("--steps", default="1-5", help="Step range: 1-5, 1,2,3, etc.")
    parser.add_argument("--video_url", default="https://www.youtube.com/watch?v=PdK9tSRqEwE&list=WL&index=7&t=95s")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not found. Pass --api_key or add to .env file.")
        sys.exit(1)

    steps = []
    for part in args.steps.split(","):
        if "-" in part:
            start, end = part.split("-")
            steps.extend(range(int(start), int(end) + 1))
        else:
            steps.append(int(part))

    step_map = {
        1: ("01_download_video.py", "Download video + extract frames", ["--url", args.video_url]),
        2: ("02_download_datasets.py", "Download ball dataset from Roboflow", ["--api_key", api_key]),
        3: ("03_train_ball_detector.py", "Train YOLOv8n ball detector", []),
        4: ("05_export_tflite.py", "Export to TFLite INT8", [
            "--weights", os.path.join("models", "ball_detector", "train", "weights", "best.pt"),
        ]),
        5: ("06_test_model.py", "Test model on video frames", [
            "--model", os.path.join("models", "ball_detector", "train", "weights", "best.pt"),
            "--source", os.path.join("data", "raw_frames"),
        ]),
    }

    for step_num in steps:
        if step_num not in step_map:
            print(f"[SKIP] Unknown step {step_num}")
            continue
        script_name, desc, step_args = step_map[step_num]
        run_step(script_name, desc, step_args)

    print(f"\n{'='*60}")
    print("[DONE] Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
