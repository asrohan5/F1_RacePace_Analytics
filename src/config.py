import os
from src.logger import logging

# ─────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────
BASE_DIR = os.getcwd()

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_SPLITS_DIR    = os.path.join(BASE_DIR, "data", "splits")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR      = os.path.join(BASE_DIR, "artifacts")
LOGS_DIR           = os.path.join(BASE_DIR, "logs")
CACHE_DIR          = os.path.join(BASE_DIR, "cache")  # FastF1 cache

# Ensure all directories exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SPLITS_DIR,
             MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────
# RAW DATA FILE PATHS  (saved by data_ingestion.py)
# ─────────────────────────────────────────
LAPS_RAW_PATH      = os.path.join(DATA_RAW_DIR, "laps_raw.csv")
TELEMETRY_RAW_PATH = os.path.join(DATA_RAW_DIR, "telemetry_raw.parquet")

# ─────────────────────────────────────────
# PROCESSED DATA FILE PATHS  (saved by data_transformation.py)
# ─────────────────────────────────────────
FEATURES_PATH      = os.path.join(DATA_PROCESSED_DIR, "features.csv")
PREPROCESSOR_PATH  = os.path.join(MODELS_DIR, "preprocessor.pkl")

FEATURES_SAME_COMPOUND_PATH = os.path.join(DATA_PROCESSED_DIR, "features_same_compound.csv")
TRAIN_SC_PATH = os.path.join(DATA_SPLITS_DIR, "train_sc.csv")
VAL_SC_PATH   = os.path.join(DATA_SPLITS_DIR, "val_sc.csv")
TEST_SC_PATH  = os.path.join(DATA_SPLITS_DIR, "test_sc.csv")

# ─────────────────────────────────────────
# SPLIT FILE PATHS  (saved by data_transformation.py)
# ─────────────────────────────────────────
TRAIN_PATH    = os.path.join(DATA_SPLITS_DIR, "train.csv")
VAL_PATH      = os.path.join(DATA_SPLITS_DIR, "val.csv")
TEST_PATH     = os.path.join(DATA_SPLITS_DIR, "test.csv")

# ─────────────────────────────────────────
# MODEL FILE PATHS  (saved by model_trainer.py)
# ─────────────────────────────────────────
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
REGRESSOR_PATH  = os.path.join(MODELS_DIR, "regressor.pkl")
CLASSIFIER_SC_PATH = os.path.join(MODELS_DIR, "classifier_sc.pkl")
REGRESSOR_SC_PATH  = os.path.join(MODELS_DIR, "regressor_sc.pkl")

# ─────────────────────────────────────────
# RACE CONFIGURATION
# ─────────────────────────────────────────
SEASON = 2021

# Races selected: Bahrain (R1), Spain (R4), Abu Dhabi (R22)
# Format: (round_number, label)
RACES = [
    (1,  "Bahrain"),
    (4,  "Spain"),
    (22, "AbuDhabi"),
]

DRIVERS = ["VER", "HAM"]

SESSION_TYPE = "R"  # Race session

# ─────────────────────────────────────────
# TELEMETRY COLUMN NAMES  (as returned by FastF1)
# ─────────────────────────────────────────
TEL_SPEED    = "Speed"
TEL_THROTTLE = "Throttle"
TEL_BRAKE    = "Brake"
TEL_GEAR     = "nGear"
TEL_RPM      = "RPM"
TEL_DISTANCE = "Distance"

# ─────────────────────────────────────────
# LAP COLUMN NAMES  (as returned by FastF1)
# ─────────────────────────────────────────
LAP_TIME_COL      = "LapTime"
LAP_NUMBER_COL    = "LapNumber"
COMPOUND_COL      = "Compound"
TYRE_LIFE_COL     = "TyreLife"
TRACK_STATUS_COL  = "TrackStatus"
PIT_OUT_COL       = "PitOutTime"
PIT_IN_COL        = "PitInTime"

# ─────────────────────────────────────────
# FEATURE ENGINEERING CONSTANTS
# ─────────────────────────────────────────

# Throttle below this threshold = not on power
THROTTLE_OFF_THRESHOLD = 5

# Throttle above this threshold = full throttle
FULL_THROTTLE_THRESHOLD = 98

# Minimum brake zone length in metres to count as a real braking zone
# (filters sensor noise / tiny blips)
MIN_BRAKE_ZONE_LENGTH_M = 20

# Coasting = Throttle < THROTTLE_OFF_THRESHOLD AND Brake == False
# No constant needed, derived from above

# ─────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT RATIOS
# ─────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# Split is done per race to avoid data leakage across rounds

# ─────────────────────────────────────────
# TARGET COLUMNS
# ─────────────────────────────────────────
TARGET_REGRESSION      = "lap_time_delta_sec"  # VER LapTime - HAM LapTime per lap (seconds)
TARGET_CLASSIFICATION  = "ver_faster"           # 1 if VER faster than HAM on that lap, 0 otherwise

# ─────────────────────────────────────────
# HYPERPARAMETERS  (After EDA)
# ─────────────────────────────────────────

INITIAL_MODEL_PARAMS = {
    # Regression: predicting lap time delta (continuous, std ~0.978s)
    # Starting with RandomForest as baseline — robust to scale, no normality assumption
    "regressor": {
        "n_estimators": 100,
        "max_depth": 5,           # shallow — we have ~142 paired laps, avoid overfit
        "min_samples_leaf": 5,
        "random_state": 42
    },
    # Classifier: predicting VER_Faster (binary, balanced 47/53)
    # Starting with LogisticRegression as baseline — interpretable, good for small data
    "classifier": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42
    }
}
# ─────────────────────────────────────────
# BEST PARAMS  (empty — auto-populated by model_trainer.py after GridSearch)
# ─────────────────────────────────────────
BEST_CLASSIFIER_PARAMS = {}
BEST_REGRESSOR_PARAMS = {'subsample': 0.6, 'reg_alpha': 0.1, 'num_leaves': 31, 'n_estimators': 50, 'min_child_samples': 20, 'max_depth': 4, 'learning_rate': 0.1}


if __name__ == "__main__":
    from src.logger import logging as log
    log.info("Config loaded successfully.")
    log.info(f"Season        : {SEASON}")
    log.info(f"Races         : {RACES}")
    log.info(f"Drivers       : {DRIVERS}")
    log.info(f"Raw data dir  : {DATA_RAW_DIR}")
    log.info(f"Models dir    : {MODELS_DIR}")
    log.info(f"Artifacts dir : {ARTIFACTS_DIR}")
    log.info(f"Split ratios  : train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    log.info(f"Targets       : regression='{TARGET_REGRESSION}', classification='{TARGET_CLASSIFICATION}'")
    log.info("All directories verified and created.")