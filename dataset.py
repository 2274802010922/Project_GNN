import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import kagglehub
import pandas as pd


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LABEL_CANDIDATES = [
    "label", "class", "class_name", "classname", "category",
    "pill", "pill_name", "drug", "drug_name", "name", "category_name"
]
FILE_CANDIDATES = [
    "image_path", "img_path", "filepath", "file_path", "path",
    "filename", "file_name", "image", "img", "file", "id", "image_id"
]


def download_dataset():
    """
    Download the mini VAIPE dataset from Kaggle.
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
    return sorted(image_paths)


def inspect_dataset_structure(dataset_path, max_depth=2, max_files_per_folder=10):
    """
    Helper to debug dataset structure when labels are hard to locate.
    """
    dataset_path = os.path.abspath(dataset_path)
    print("=== FOLDER TREE ===")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, "").count(os.sep)
        if level > max_depth:
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for file_name in files[:max_files_per_folder]:
            print(f"{indent}  - {file_name}")

    print("\n=== CANDIDATE ANNOTATION FILES ===")
    found = False
    for ext in ("*.csv", "*.json", "*.txt", "*.xml"):
        for p in Path(dataset_path).rglob(ext):
            found = True
            print(p)
    if not found:
        print("No CSV/JSON/TXT/XML files found.")


def _normalize_string(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None
    return value.replace("\\", "/")


def _build_image_lookup(image_paths, dataset_path):
    lookup = {}
    dataset_path = os.path.abspath(dataset_path)
    for img_path in image_paths:
        rel = os.path.relpath(img_path, dataset_path).replace("\\", "/")
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        candidates = {
            img_path,
            os.path.abspath(img_path),
            rel,
            basename,
            stem,
            rel.lower(),
            basename.lower(),
            stem.lower(),
        }
        for key in candidates:
            lookup[key] = img_path
    return lookup


def _candidate_column(columns, priorities):
    lowered = {str(col).strip().lower(): col for col in columns}
    for key in priorities:
        if key in lowered:
            return lowered[key]
    return None


def _match_image_path(raw_value, image_lookup):
    value = _normalize_string(raw_value)
    if value is None:
        return None

    basename = os.path.basename(value)
    stem = os.path.splitext(basename)[0]

    for key in [value, value.lower(), basename, basename.lower(), stem, stem.lower()]:
        if key in image_lookup:
            return image_lookup[key]
    return None


def _finalize_mapping(mapping, image_paths):
    if not mapping:
        return None

    coverage = len(mapping) / max(len(image_paths), 1)
    labels = [mapping.get(path) for path in image_paths]
    known_labels = [label for label in labels if label is not None]
    unique_labels = sorted(set(known_labels))

    if coverage < max(0.25, 2 / max(len(image_paths), 1)):
        return None
    if len(unique_labels) < 2:
        return None

    fallback = {}
    if known_labels:
        majority = Counter(known_labels).most_common(1)[0][0]
        for path in image_paths:
            fallback[path] = mapping.get(path, majority)

    completed_labels = [fallback[path] for path in image_paths]
    return completed_labels, coverage, unique_labels


def _try_load_labels_from_csv(dataset_path, image_paths):
    csv_files = sorted(Path(dataset_path).rglob("*.csv"))
    if not csv_files:
        return None, None

    image_lookup = _build_image_lookup(image_paths, dataset_path)
    best_result = None
    best_coverage = -1.0

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue

        file_col = _candidate_column(df.columns, FILE_CANDIDATES)
        label_col = _candidate_column(df.columns, LABEL_CANDIDATES)

        if file_col is None:
            possible_cols = [c for c in df.columns if "image" in str(c).lower() or "file" in str(c).lower()]
            file_col = possible_cols[0] if possible_cols else None
        if label_col is None:
            possible_cols = [c for c in df.columns if "label" in str(c).lower() or "class" in str(c).lower()]
            label_col = possible_cols[0] if possible_cols else None

        if file_col is None or label_col is None:
            continue

        mapping = {}
        for _, row in df.iterrows():
            matched_path = _match_image_path(row[file_col], image_lookup)
            label_value = _normalize_string(row[label_col])
            if matched_path is not None and label_value is not None:
                mapping[matched_path] = label_value

        result = _finalize_mapping(mapping, image_paths)
        if result is None:
            continue

        labels, coverage, _ = result
        if coverage > best_coverage:
            best_coverage = coverage
            best_result = (labels, f"csv:{csv_path}")

    return best_result if best_result is not None else (None, None)


def _flatten_json_records(obj):
    records = []

    def _walk(item):
        if isinstance(item, dict):
            if any(k.lower() in FILE_CANDIDATES for k in item.keys()) or any(k.lower() in LABEL_CANDIDATES for k in item.keys()):
                records.append(item)
            for value in item.values():
                _walk(value)
        elif isinstance(item, list):
            for value in item:
                _walk(value)

    _walk(obj)
    return records


def _try_load_labels_from_json(dataset_path, image_paths):
    json_files = sorted(Path(dataset_path).rglob("*.json"))
    if not json_files:
        return None, None

    image_lookup = _build_image_lookup(image_paths, dataset_path)
    best_result = None
    best_coverage = -1.0

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception:
            continue

        records = _flatten_json_records(content)
        if not records:
            continue

        mapping = {}
        for record in records:
            file_key = None
            label_key = None
            lowered = {str(k).lower(): k for k in record.keys()}

            for candidate in FILE_CANDIDATES:
                if candidate in lowered:
                    file_key = lowered[candidate]
                    break
            for candidate in LABEL_CANDIDATES:
                if candidate in lowered:
                    label_key = lowered[candidate]
                    break

            if file_key is None or label_key is None:
                continue

            matched_path = _match_image_path(record[file_key], image_lookup)
            label_value = _normalize_string(record[label_key])
            if matched_path is not None and label_value is not None:
                mapping[matched_path] = label_value

        result = _finalize_mapping(mapping, image_paths)
        if result is None:
            continue

        labels, coverage, _ = result
        if coverage > best_coverage:
            best_coverage = coverage
            best_result = (labels, f"json:{json_path}")

    return best_result if best_result is not None else (None, None)


def _try_load_labels_from_txt(dataset_path, image_paths):
    txt_files = sorted(Path(dataset_path).rglob("*.txt"))
    if not txt_files:
        return None, None

    image_lookup = _build_image_lookup(image_paths, dataset_path)
    best_result = None
    best_coverage = -1.0

    for txt_path in txt_files:
        mapping = {}
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = re.split(r"[\t,; ]+", line)
                    if len(parts) < 2:
                        continue
                    matched_path = _match_image_path(parts[0], image_lookup)
                    label_value = _normalize_string(parts[1])
                    if matched_path is not None and label_value is not None:
                        mapping[matched_path] = label_value
        except Exception:
            continue

        result = _finalize_mapping(mapping, image_paths)
        if result is None:
            continue

        labels, coverage, _ = result
        if coverage > best_coverage:
            best_coverage = coverage
            best_result = (labels, f"txt:{txt_path}")

    return best_result if best_result is not None else (None, None)


def _try_load_labels_from_xml(dataset_path, image_paths):
    xml_files = sorted(Path(dataset_path).rglob("*.xml"))
    if not xml_files:
        return None, None

    image_lookup = _build_image_lookup(image_paths, dataset_path)
    best_result = None
    best_coverage = -1.0
    mapping = {}

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue

        filename_tag = root.find(".//filename")
        name_tag = root.find(".//object/name")

        if filename_tag is None or name_tag is None:
            continue

        matched_path = _match_image_path(filename_tag.text, image_lookup)
        label_value = _normalize_string(name_tag.text)
        if matched_path is not None and label_value is not None:
            mapping[matched_path] = label_value

    result = _finalize_mapping(mapping, image_paths)
    if result is not None:
        labels, coverage, _ = result
        best_result = (labels, "xml")
        best_coverage = coverage

    return best_result if best_result is not None else (None, None)


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


def load_image_paths_and_labels(dataset_path, require_multiple_classes=True):
    """
    Load image paths and labels.
    Priority: CSV -> JSON -> TXT -> XML -> folder names.
    """
    image_paths = _find_all_images(dataset_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {dataset_path}")

    source = None
    labels = None

    for loader in (
        _try_load_labels_from_csv,
        _try_load_labels_from_json,
        _try_load_labels_from_txt,
        _try_load_labels_from_xml,
    ):
        labels, source = loader(dataset_path, image_paths)
        if labels is not None:
            break

    if labels is None:
        labels, source = _infer_labels_from_folders(image_paths, dataset_path)

    class_names = sorted(set(labels))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = [class_to_idx[label] for label in labels]

    counts = Counter(labels)
    print(f"Label source: {source}")
    print(f"Total images: {len(image_paths)}")
    print(f"Total classes: {len(class_names)}")
    print("Top classes by sample count:")
    for class_name, count in counts.most_common(15):
        print(f"  {class_name}: {count}")

    if require_multiple_classes and len(class_names) <= 1:
        candidate_files = list(Path(dataset_path).rglob("*.csv")) + list(Path(dataset_path).rglob("*.json")) + list(Path(dataset_path).rglob("*.txt")) + list(Path(dataset_path).rglob("*.xml"))
        candidate_files = [str(p) for p in sorted(candidate_files)[:20]]
        message = [
            "Only 1 class was discovered, so few-shot classification cannot run correctly.",
            f"Current label source: {source}",
            "This usually means the dataset stores all images in one folder and the real labels are in an annotation file that was not matched.",
        ]
        if candidate_files:
            message.append("Candidate annotation files found:")
            message.extend(candidate_files)
        message.append("Run inspect_dataset_structure(dataset_path) to inspect the dataset tree.")
        raise ValueError("\n".join(message))

    return image_paths, y, class_names, source
