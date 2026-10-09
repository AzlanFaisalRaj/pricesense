"""Paths and settings used by every other file."""
from pathlib import Path

# the project root is the folder that contains "src"
ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / 'data' / 'raw' / 'pakwheels_pakistan_automobile_dataset.csv'
PROCESSED_DATA = ROOT / 'data' / 'processed' / 'cleaned_cars.csv'

MODEL_DIR = ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'model.joblib'
METADATA_PATH = MODEL_DIR / 'model_metadata.json'

# The listings were scraped in 2024, so car age is counted from this year.
# The app must use the same value, otherwise the model sees a different age than it was trained on.
REF_YEAR = 2024

RANDOM_STATE = 42
