import os
import sys
import yaml
import argparse
from dotenv import load_dotenv

load_dotenv()


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "datasets.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", help="Roboflow API key (or set ROBOFLOW_API_KEY env)")
    parser.add_argument("--output_dir", default="data/ball_dataset")
    args = parser.parse_args()
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not found. Pass --api_key or set ROBOFLOW_API_KEY env var.")
        sys.exit(1)

    cfg = load_config()
    ball_cfg = cfg["ball_detection"]

    import roboflow
    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace(ball_cfg["workspace"]).project(ball_cfg["project"])
    version = project.version(ball_cfg["version"])

    base = os.path.join(os.path.dirname(__file__), "..")
    output_dir = os.path.join(base, args.output_dir)
    dataset = version.download(
        model_format=ball_cfg["format"],
        location=output_dir,
        overwrite=False,
    )
    print(f"[DONE] Dataset downloaded to: {dataset.location}")
    print(f"Classes: {ball_cfg['classes']}")


if __name__ == "__main__":
    main()
