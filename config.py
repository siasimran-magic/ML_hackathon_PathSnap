import os
from pathlib import Path

# Base directories
BASE_DIR = Path(r"C:\Users\siasi\Downloads\Student folder")
DATA_DIR = BASE_DIR / "Student folder"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "student_test"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for d in [OUTPUT_DIR, EDA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42

# Geodesic constants
EARTH_RADIUS_METERS = 6371000.0

# Candidate search parameters
CANDIDATE_SEARCH_RADIUS_METERS = 40.0
CANDIDATE_CELL_SIZE_METERS = 50.0
MAX_CANDIDATES_PER_OBS = 8
MIN_CANDIDATES_PER_OBS = 3

# Nominal sampling frequency
NOMINAL_SAMPLING_HZ = 1.0
GAP_THRESHOLD_SECONDS = 2.0
LARGE_GAP_THRESHOLD_SECONDS = 5.0
