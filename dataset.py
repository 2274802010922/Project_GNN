import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import kagglehub
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def download_dataset():
    """Download the mini VAIPE dataset from Kaggle."""
    path = kagglehub.dataset_download("anhduy091100/vaipe-minimal-dataset")
    print(f"Dataset downloaded to: {path}")
    return path


def inspect_dataset_structure(dataset_path, max_depth=3, max_files_per_folder=10):
    """Print a compact folder tree and candidate annotation files."""
    dataset_path = os.path.abspath(dataset_path)
    print("=== TREE (max depth = %d) ===" % max_depth)
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, "").count(os.sep)
        if level > max_depth:
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:max_files_per_folder]:
            print(f"{indent}  - {f}")

    print("\n=== POSSIBLE LABEL FILES ===")
    found = False
    for ext in ("*.csv", "*.json", "*.txt", "*.xml"):
        for p in Path(dataset_path).rglob(ext):
            found = True
            print(p)
    if not found:
        print("No annotation files found.")


# ---------------------------------------------------------------------
# VAIPE-specific helpers
# ---------------------------------------------------------------------

def _find_vaipe_roots(dataset_path):
    base = Path(dataset_path)
    image_dir = None
    label_dir = None

    for p in base.rglob("images"):
        if p.is_dir() and p.parent.name == "pills":
            image_dir = p
            break

    for p in base.rglob("labels"):
        if p.is_dir() and p.parent.name == "pills":
            label_dir = p
            break

    return image_dir, label_dir


def _safe_int(v, default=0):
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _sanitize_label(label):
    return str(label).strip().replace("/", "-") if label is not None else "unknown"


def _crop_single_box(img, box):
    x = _safe_int(box.get("x", 0))
    y = _safe_int(box.get("y", 0))
    w = _safe_int(box.get("w", 0))
    h = _safe_int(box.get("h", 0))

    if w <= 1 or h <= 1:
        return None

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img.width, x + w)
    y2 = min(img.height, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return img.crop((x1, y1, x2, y2))


def _get_writable_crop_root():
    """
    Choose a writable directory for generated crops.

    We intentionally avoid saving inside `dataset_path` because KaggleHub can
    return read-only locations such as `/kaggle/input/...`.
    """
    candidates = [
        Path("/content/generated_crops_fewshot"),
        Path.cwd() / "generated_crops_fewshot",
        Path("/tmp/generated_crops_fewshot"),
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return candidate
        except Exception:
            continue

    raise OSError(
        "Could not find a writable directory for generated crops. "
        "Tried /content, current working directory, and /tmp."
    )


def _load_vaipe_cropped_instances(dataset_path):
    image_dir, label_dir = _find_vaipe_roots(dataset_path)
    if image_dir is None or label_dir is None:
        return None

    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        return None

    crop_root = _get_writable_crop_root()
    print(f"Crop output directory: {crop_root}")

    sample_paths = []
    sample_labels = []
    samples_per_image = defaultdict(int)

    for json_path in label_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except Exception:
            continue

        image_path = None

        # Priority 1: use `path` field if present
        raw_path = ann.get("path") if isinstance(ann, dict) else None
        if raw_path:
            raw_name = os.path.basename(str(raw_path))
            cand = image_dir / raw_name
            if cand.exists():
                image_path = cand

        # Priority 2: same stem as JSON file
        if image_path is None:
            stem = json_path.stem
            for ext in IMAGE_EXTENSIONS:
                cand = image_dir / f"{stem}{ext}"
                if cand.exists():
                    image_path = cand
                    break

        if image_path is None or not image_path.exists():
            continue

        boxes = ann.get("boxes", []) if isinstance(ann, dict) else []
        if not isinstance(boxes, list) or len(boxes) == 0:
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        for idx, box in enumerate(boxes):
            if not isinstance(box, dict):
                continue

            label = _sanitize_label(box.get("label"))
            crop = _crop_single_box(img, box)
            if crop is None:
                continue

            class_dir = crop_root / label
            class_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{json_path.stem}_box{idx}.jpg"
            out_path = class_dir / out_name

            # Save once; reuse on future runs.
            if not out_path.exists():
                crop.save(out_path, quality=95)

            sample_paths.append(str(out_path))
            sample_labels.append(label)
            samples_per_image[str(image_path)] += 1

    if not sample_paths:
        return None

    return sample_paths, sample_labels, "vaipe_json_boxes"


# ---------------------------------------------------------------------
# Generic folder fallback
# ---------------------------------------------------------------------

def _find_all_images(dataset_path):
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, file_name))
    return sorted(image_paths)


def _infer_labels_from_folders(image_paths, dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    labels = []
    for img_path in image_paths:
        rel_parts = os.path.relpath(img_path, dataset_path).split(os.sep)
        if len(rel_parts) >= 2:
            labels.append(rel_parts[-2])
        else:
            labels.append(os.path.basename(os.path.dirname(img_path)))
    return labels, "folder"


# ---------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------

def load_image_paths_and_labels(dataset_path, require_multiple_classes=True, min_samples_per_class=2):
    """
    Load mini VAIPE as cropped pill instances.

    For this dataset, labels are not stored as one label per whole image.
    They are stored in `pills/labels/*.json`, where each JSON contains
    multiple bounding boxes and each box has a `label` field.

    This function converts each annotated box into one cropped image sample.
    """
    vaipe_result = _load_vaipe_cropped_instances(dataset_path)

    if vaipe_result is not None:
        image_paths, raw_labels, source = vaipe_result
    else:
        image_paths = _find_all_images(dataset_path)
        if not image_paths:
            raise FileNotFoundError(f"No images found under: {dataset_path}")
        raw_labels, source = _infer_labels_from_folders(image_paths, dataset_path)

    counts = Counter(raw_labels)

    # Keep only classes with enough samples for at least 1-shot/1-query few-shot episodes.
    kept_classes = {cls for cls, cnt in counts.items() if cnt >= min_samples_per_class}
    filtered_paths = []
    filtered_labels = []
    for p, lbl in zip(image_paths, raw_labels):
        if lbl in kept_classes:
            filtered_paths.append(p)
            filtered_labels.append(lbl)

    if not filtered_paths:
        raise ValueError(
            "No valid few-shot samples were created. This usually means the JSON annotations "
            "could not be matched to images or every class has too few samples."
        )

    class_names = sorted(set(filtered_labels))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = [class_to_idx[label] for label in filtered_labels]

    final_counts = Counter(filtered_labels)
    print(f"Label source: {source}")
    print(f"Total cropped instances: {len(filtered_paths)}")
    print(f"Total classes after filtering (>= {min_samples_per_class} samples/class): {len(class_names)}")
    print("Top classes by sample count:")
    for class_name, count in final_counts.most_common(20):
        print(f"  {class_name}: {count}")

    if require_multiple_classes and len(class_names) <= 1:
        raise ValueError(
            "Only 1 class was discovered after reading VAIPE JSON boxes. "
            "Few-shot classification cannot run correctly."
        )

    return filtered_paths, y, class_names, source
