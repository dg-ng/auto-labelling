import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = REPO_ROOT / "embeddings_cache"
RESULTS_DIR = REPO_ROOT / "results"

MASTER_DATA_PATH = DATA_DIR / "master_data.csv"

# 16 balanced categories in data/master_data.csv, alphabetical so the
# label<->name mapping is deterministic and independent of CSV row order.
CLASS_NAMES = [
    "ARTS & CULTURE", "BUSINESS", "COMEDY", "CRIME", "EDUCATION",
    "ENTERTAINMENT", "ENVIRONMENT", "HEALTH", "MEDIA", "NEWS",
    "POLITICS", "RELIGION", "SCIENCE", "SPORTS", "TECH", "WOMEN",
]
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42
LABEL_FRACTION = 0.05
TEST_FRACTION = 0.20  # stratified train/test split of master_data.csv
MIN_WORDS = 5  # drop degenerate rows (e.g. a body of just "(CNN)") below this word count

# master_data.csv is only ~4,750 rows after cleaning (vs. AG News's
# 120,000) — small enough that every method runs on the FULL dataset.
# None = no cap. Kept as tunable knobs (not deleted) in case a future
# dev-scale iteration needs fast sampling again.
SAMPLE_SIZE = None
ROBERTA_SAMPLE_SIZE = None
CLASSIFIER_SAMPLE_SIZE = None

# silhouette_score/davies_bouldin_score are O(n^2) in the number of points
# — sub-sample before computing them now that clustering runs full-data.
SILHOUETTE_SAMPLE_SIZE = 2000

CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"

# Additional semi-supervised model (mentor feedback item 1) — compared
# *alongside* CLASSIFIER_MODEL_NAME, not replacing it.
CLASSIFIER_MODEL_NAME_ALT = "google/electra-small-discriminator"

# Summarization model feeding the unsupervised track (raw text -> summary
# sentence -> embed -> cluster). Swapped from the distilled
# sshleifer/distilbart-cnn-6-6 to the full BART-large model it was
# distilled from — more fluent, coherent multi-sentence summaries, at the
# cost of slower per-row CPU inference (a one-time cost, cached to disk
# via train_clean/test_clean's saved `summary` column).
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"
