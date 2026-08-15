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
ROBERTA_SAMPLE_SIZE = 500  # separate, smaller cap for CPU-bound frozen RoBERTa embedding; None = full data
CLASSIFIER_SAMPLE_SIZE = 150  # separate, smaller cap for DistilBERT fine-tuning (tasks 11, 12); None = full data
# Measured on this machine: fine-tuning throughput is ~0.54s/row/epoch and
# frozen-model inference is ~0.12s/row (CPU). At 500 rows, task 11's 4
# fine-tune calls + prediction passes exceeded a 3600s cell timeout; 150
# keeps task 11 comfortably under budget with margin. See Task 11's plan note.
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"