import os
from collections import Counter
from pathlib import Path

import kagglehub
import pandas as pd


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def download_dataset():
    """
    Download the public mini VAIPE dataset from Kaggle.
    Returns the local extracted folder path.
    """
    path = kagglehub.dataset_download("anhduy091100/vaipe-minimal-dataset")
    print(f"Dataset downloaded to: {path}")
    return path


def _find_all_images(dataset_path):
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, file_name))
    image_paths = sorted(image_paths)
    return image_paths


def _candidate_label_columns(columns):
    priority = [
        "label", "class", "class_name", "classname", "category",
        "pill", "pill_name", "drug", "drug_name", "name"
    ]
    lowered = {col.lower(): col for col in columns}
    for key in priority:
        if key in lowered:
            return lowered[key]
    return None


def _candidate_file_columns(columns):
    priority = [
        "image_path", "img_path", "filepath", "file_path", "path",
        "filename", "file_name", "image", "img", "file"
    ]
    lowered = {col.lower(): col for col in columns}
    for key in priority:
        if key in lowered:
            return lowered[key]
    return None


def _try_load_labels_from_csv(dataset_path, image_paths):
    csv_files = sorted(Path(dataset_path).rglob("*.csv"))
    if not csv_files:
        return None

    basename_to_path = {}
    for img_path in image_paths:
        basename_to_path.setdefault(os.path.basename(img_path), img_path)

    best_mapping = None
    best_coverage = 0.0

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty:
            continue

        file_col = _candidate_file_columns(df.columns)
        label_col = _candidate_label_columns(df.columns)
        if file_col is None or label_col is None:
            continue

        mapping = {}
        for _, row in df.iterrows():
            file_value = str(row[file_col]).strip()
            label_value = str(row[label_col]).strip()
            if not file_value or not label_value or label_value.lower() == "nan":
                continue

            candidate_keys = [
                file_value,
                os.path.basename(file_value),
                file_value.replace("\\", "/").split("/")[-1],
            ]

            matched_path = None
            for key in candidate_keys:
                if key in basename_to_path:
                    matched_path = basename_to_path[key]
                    break

            if matched_path is not None:
                mapping[matched_path] = label_value

        coverage = len(mapping) / max(len(image_paths), 1)
        if coverage > best_coverage:
            best_mapping = mapping
            best_coverage = coverage

    if best_mapping is not None and best_coverage >= 0.5:
        print(f"Loaded labels from CSV annotations with coverage: {best_coverage:.2%}")
        labels = [best_mapping.get(p, os.path.basename(os.path.dirname(p))) for p in image_paths]
        return labels

    return None


def _infer_labels_from_folders(image_paths, dataset_path):
    """
    Fallback: use the parent folder name as the class label.
    This works well for datasets organized as .../<class_name>/<image>.
    """
    labels = []
    dataset_path = os.path.abspath(dataset_path)

    for img_path in image_paths:
        rel_parts = os.path.relpath(img_path, dataset_path).split(os.sep)

        if len(rel_parts) >= 2:
            label = rel_parts[-2]
        else:
            label = os.path.basename(os.path.dirname(img_path))

        labels.append(label)

    return labels


def load_image_paths_and_labels(dataset_path):
    image_paths = _find_all_images(dataset_path)
    if not image_paths:
        raise FileNotFoundError(f"No image files found under: {dataset_path}")

    labels = _try_load_labels_from_csv(dataset_path, image_paths)
    if labels is None:
        labels = _infer_labels_from_folders(image_paths, dataset_path)
        print("Labels inferred from folder names.")

    class_names = sorted(set(labels))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = [class_to_idx[label] for label in labels]

    counts = Counter(labels)
    print(f"Total images: {len(image_paths)}")
    print(f"Total classes: {len(class_names)}")
    print("Top classes by sample count:")
    for class_name, count in counts.most_common(10):
        print(f"  {class_name}: {count}")

    return image_paths, y, class_names
