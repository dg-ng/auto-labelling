import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = REPO_ROOT / "embeddings_cache"
RESULTS_DIR = REPO_ROOT / "results"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = 4
SEED = 42
LABEL_FRACTION = 0.05
SAMPLE_SIZE = 8000  # train-set dev cap; set to None for the full-data final run
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
