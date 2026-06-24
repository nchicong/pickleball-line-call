import subprocess
import argparse
import os
import cv2
from pathlib import Path
from tqdm import tqdm


def download_video(url: str, output_path: str) -> str:
    video_path = os.path.join(output_path, "source_video.mp4")
    if os.path.exists(video_path):
        print(f"[SKIP] Video already exists: {video_path}")
        return video_path
    print(f"[DOWNLOAD] {url} -> {video_path}")
    subprocess.run(
        ["yt-dlp", "-f", "best[height<=720]", "-o", video_path, url],
        check=True,
    )
    return video_path


def extract_frames(video_path: str, output_dir: str, fps: float = 5.0):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > fps else 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved_idx = 0
    pbar = tqdm(total=total_frames, desc="Extracting frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"[DONE] {saved_idx} frames extracted to {output_dir}")
    return saved_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.youtube.com/watch?v=PdK9tSRqEwE&list=WL&index=7&t=95s")
    parser.add_argument("--output_dir", default="data/raw_video")
    parser.add_argument("--frames_dir", default="data/raw_frames")
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()
    base = os.path.join(os.path.dirname(__file__), "..")
    output_dir = os.path.join(base, args.output_dir)
    frames_dir = os.path.join(base, args.frames_dir)
    video_path = download_video(args.url, output_dir)
    extract_frames(video_path, frames_dir, args.fps)


if __name__ == "__main__":
    main()
