import os
import yaml
import argparse
from ultralytics import YOLO


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "datasets.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_data_yaml(dataset_dir: str) -> str:
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f == "data.yaml":
                return os.path.join(root, f)
    raise FileNotFoundError(f"data.yaml not found under {dataset_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--dataset_dir", default="data/ball_dataset")
    parser.add_argument("--output_dir", default="models/ball_detector/tflite")
    args = parser.parse_args()

    cfg = load_config()
    export_cfg = cfg["export"]
    base = os.path.join(os.path.dirname(__file__), "..")
    dataset_dir = os.path.join(base, args.dataset_dir)
    output_dir = os.path.join(base, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_yaml = find_data_yaml(dataset_dir)
    print(f"[INFO] Loading model: {args.weights}")
    model = YOLO(args.weights)

    print("[INFO] Exporting to TFLite INT8...")
    model.export(
        format="tflite",
        int8=export_cfg["int8"],
        data=data_yaml,
        imgsz=export_cfg["imgsz"],
    )

    saved_model_dir = args.weights.replace("best.pt", "") + f"{Path(args.weights).stem}_saved_model"
    import glob
    tflite_files = glob.glob(os.path.join(os.path.dirname(saved_model_dir), "*_saved_model", "*full_integer_quant.tflite"))
    if tflite_files:
        src = tflite_files[0]
        dst = os.path.join(output_dir, "ball_model_full_integer_quant.tflite")
        import shutil
        shutil.copy2(src, dst)
        print(f"[DONE] TFLite model exported to: {dst}")
    else:
        print(f"[WARN] TFLite file not found. Check {saved_model_dir}")

    from pathlib import Path


if __name__ == "__main__":
    main()
