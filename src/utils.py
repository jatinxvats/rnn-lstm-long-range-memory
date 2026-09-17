from pathlib import Path

def get_project_root() -> Path:
    # this file lives at <repo_root>/src/utils.py, so parent.parent is repo root
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"