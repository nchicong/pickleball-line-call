import os, shutil, glob, random
from collections import defaultdict


def merge_datasets():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    out_dir = os.path.join(base, "ball_dataset_v2")
    sources = [
        ("ball_dataset_uninu", "uninu"),
        ("ball_dataset_pball", "pball"),
    ]

    random.seed(42)

    splits = ["train", "valid", "test"]
    for sp in splits:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(out_dir, sp, sub), exist_ok=True)

    total_counts = {sp: {"images": 0, "labels": 0, "ball_anns": 0} for sp in splits}

    for src_dir, prefix in sources:
        for sp in splits:
            src_img_dir = os.path.join(base, src_dir, sp, "images")
            src_lbl_dir = os.path.join(base, src_dir, sp, "labels")

            if not os.path.exists(src_img_dir):
                print(f"  [SKIP] {src_dir}/{sp} — not found")
                continue

            for img_file in sorted(glob.glob(os.path.join(src_img_dir, "*.*"))):
                basename = os.path.splitext(os.path.basename(img_file))[0]
                ext = os.path.splitext(img_file)[1]
                new_basename = f"{prefix}_{basename}"

                # Copy image
                dst_img = os.path.join(out_dir, sp, "images", new_basename + ext)
                shutil.copy2(img_file, dst_img)

                # Copy label (only ball class, class_id 0)
                src_label = os.path.join(src_lbl_dir, basename + ".txt")
                if os.path.exists(src_label):
                    dst_label = os.path.join(out_dir, sp, "labels", new_basename + ".txt")
                    kept_lines = []
                    with open(src_label) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5 and int(parts[0]) == 0:
                                kept_lines.append(line.strip())
                    if kept_lines:
                        with open(dst_label, "w") as f:
                            f.write("\n".join(kept_lines) + "\n")
                        total_counts[sp]["ball_anns"] += len(kept_lines)
                    else:
                        # Image has no ball annotations — create empty label
                        with open(dst_label, "w") as f:
                            pass
                else:
                    # No label at all
                    dst_label = os.path.join(out_dir, sp, "labels", new_basename + ".txt")
                    with open(dst_label, "w"):
                        pass

                total_counts[sp]["images"] += 1
                total_counts[sp]["labels"] += 1

    # Create data.yaml
    # Count images with actual ball annotations for stats
    for sp in splits:
        lbl_dir = os.path.join(out_dir, sp, "labels")
        with_ball = sum(
            1 for lf in glob.glob(os.path.join(lbl_dir, "*.txt"))
            if os.path.getsize(lf) > 0
        )
        total_counts[sp]["with_ball"] = with_ball

    data_yaml = {
        "names": ["ball"],
        "nc": 1,
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
    }
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        import yaml
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\n=== Merge Summary ===")
    for sp in splits:
        c = total_counts[sp]
        print(f"  {sp}: {c['images']} images, {c['labels']} labels, {c['ball_anns']} ball anns, {c.get('with_ball',0)} with ball")
    total_imgs = sum(c["images"] for c in total_counts.values())
    total_ball = sum(c["ball_anns"] for c in total_counts.values())
    print(f"  TOTAL: {total_imgs} images, {total_ball} ball annotations")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    merge_datasets()
