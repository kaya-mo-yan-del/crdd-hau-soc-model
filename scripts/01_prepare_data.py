import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs import config
from src.data.split_dataset import create_data_splits

if __name__ == "__main__":
    print("Splitting data...")
    create_data_splits(config.DATA_DIR, config.SPLITS_DIR, config.CLASSES)
