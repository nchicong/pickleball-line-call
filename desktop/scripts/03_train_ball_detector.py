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
    parser.add_argument("--dataset_dir", default="data/ball_dataset")
    parser.add_argument("--output_dir", default="models/ball_detector")
    args = parser.parse_args()

    cfg = load_config()
    train_cfg = cfg["training"]["ball_detector"]

    base = os.path.join(os.path.dirname(__file__), "..")
    dataset_dir = os.path.join(base, args.dataset_dir)
    output_dir = os.path.join(base, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_yaml = find_data_yaml(dataset_dir)
    print(f"[INFO] Using data.yaml: {data_yaml}")

    last_ckpt = os.path.join(output_dir, "train", "weights", "last.pt")
    if os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from checkpoint: {last_ckpt}")
        model = YOLO(last_ckpt)
        results = model.train(
            resume=True,
            workers=2,
            device=0,
            verbose=True,
        )
    else:
        model = YOLO(train_cfg["model"])
        results = model.train(
            data=data_yaml,
            epochs=train_cfg["epochs"],
            imgsz=train_cfg["imgsz"],
            batch=train_cfg["batch"],
            patience=train_cfg["patience"],
            lr0=train_cfg["lr0"],
            augment=train_cfg["augment"],
            close_mosaic=train_cfg["close_mosaic"],
            project=output_dir,
            name="train",
            device=0,
            workers=2,
            verbose=True,
        )
    print(f"[DONE] Training complete. Best model: {os.path.join(output_dir, 'train', 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
