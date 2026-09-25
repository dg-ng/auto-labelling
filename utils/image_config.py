# utils/image_config.py
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

IMAGE_DATA_DIR = DATA_DIR / "image-data"
IMAGE_MANIFEST_PATH = DATA_DIR / "image_manifest.csv"
IMAGE_EMBEDDINGS_PATH = DATA_DIR / "image_embeddings.npy"

# Alphabetical so label_idx is deterministic and independent of filesystem order
IMAGE_CLASS_NAMES = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel",
]
IMAGE_NUM_CLASSES = len(IMAGE_CLASS_NAMES)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

IMAGE_SEED = 42
IMAGE_LABEL_FRACTION = 0.05   # ~27–28 images/class as labeled seed
