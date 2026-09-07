import random
from pathlib import Path
from configs.config import SEED


def create_data_splits(data_dir: Path, output_dir: Path, classes: list):
    """Generates train/val/test split files (path,label per line).

    Note: this is a per-file split. Clips from the same bird/session can
    end up in both train and val, which can inflate validation accuracy.
    A group-based (device/session-level) split avoids this.
    """
    random.seed(SEED)

    train_paths, val_paths, test_paths = [], [], []

    for cls_idx, cls_name in enumerate(classes):
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            print(f"Warning: {cls_dir} does not exist.")
            continue

        files = sorted(cls_dir.glob("*.wav"))  # sort before shuffle for reproducibility
        random.shuffle(files)

        # 70% Train, 15% Val, 15% Test (stratified per class)
        train_split = int(0.7 * len(files))
        val_split = int(0.85 * len(files))

        train_paths.extend([(f, cls_idx) for f in files[:train_split]])
        val_paths.extend([(f, cls_idx) for f in files[train_split:val_split]])
        test_paths.extend([(f, cls_idx) for f in files[val_split:]])

        print(f"{cls_name}: {len(files)} files "
              f"({train_split} train / {val_split - train_split} val / "
              f"{len(files) - val_split} test)")

    _write_split(output_dir / "train.txt", train_paths)
    _write_split(output_dir / "val.txt", val_paths)
    _write_split(output_dir / "test.txt", test_paths)
    print("Dataset splits created successfully.")


def _write_split(filepath, data):
    with open(filepath, "w") as f:
        for path, label in data:
            f.write(f"{path},{label}\n")
