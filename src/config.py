import os

# ─────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_SPLITS_DIR    = os.path.join(BASE_DIR, "data", "splits")
MODELS_DIR         = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR      = os.path.join(BASE_DIR, "artifacts")
LOGS_DIR           = os.path.join(BASE_DIR, "logs")
CACHE_DIR          = os.path.join(BASE_DIR, "cache")

for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SPLITS_DIR,
             MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────
# RAW DATA FILE PATHS
# ─────────────────────────────────────────
LAPS_RAW_PATH      = os.path.join(DATA_RAW_DIR, "laps_raw.csv")
TELEMETRY_RAW_PATH = os.path.join(DATA_RAW_DIR, "telemetry_raw.parquet")

# ─────────────────────────────────────────
# PROCESSED DATA FILE PATHS
# ─────────────────────────────────────────
FEATURES_PATH               = os.path.join(DATA_PROCESSED_DIR, "features.csv")
FEATURES_SAME_COMPOUND_PATH = os.path.join(DATA_PROCESSED_DIR, "features_same_compound.csv")
PREPROCESSOR_PATH           = os.path.join(MODELS_DIR, "preprocessor.pkl")

# ─────────────────────────────────────────
# SPLIT FILE PATHS
# ─────────────────────────────────────────
TRAIN_PATH    = os.path.join(DATA_SPLITS_DIR, "train.csv")
VAL_PATH      = os.path.join(DATA_SPLITS_DIR, "val.csv")
TEST_PATH     = os.path.join(DATA_SPLITS_DIR, "test.csv")
TRAIN_SC_PATH = os.path.join(DATA_SPLITS_DIR, "train_sc.csv")
VAL_SC_PATH   = os.path.join(DATA_SPLITS_DIR, "val_sc.csv")
TEST_SC_PATH  = os.path.join(DATA_SPLITS_DIR, "test_sc.csv")

# ─────────────────────────────────────────
# MODEL FILE PATHS
# ─────────────────────────────────────────
CLASSIFIER_PATH    = os.path.join(MODELS_DIR, "classifier.pkl")
REGRESSOR_PATH     = os.path.join(MODELS_DIR, "regressor.pkl")
CLASSIFIER_SC_PATH = os.path.join(MODELS_DIR, "classifier_sc.pkl")
REGRESSOR_SC_PATH  = os.path.join(MODELS_DIR, "regressor_sc.pkl")

# ─────────────────────────────────────────
# SEASON CONFIGURATION
# ─────────────────────────────────────────
SEASON = 2021

# All 22 rounds of 2021 with labels
# Format: (round_number, label)
RACES = [
    (1,  "Bahrain"),
    (2,  "Imola"),
    (3,  "Portugal"),
    (4,  "Spain"),
    (5,  "Monaco"),
    (6,  "Azerbaijan"),
    (7,  "France"),
    (8,  "Styria"),
    (9,  "Austria"),
    (10, "Britain"),
    (11, "Hungary"),
    (12, "Belgium"),
    (13, "Netherlands"),
    (14, "Italy"),
    (15, "Russia"),
    (16, "Turkey"),
    (17, "UnitedStates"),
    (18, "Mexico"),
    (19, "Brazil"),
    (20, "Qatar"),
    (21, "SaudiArabia"),
    (22, "AbuDhabi"),
]

# Primary drivers — lap time delta comparison
DRIVERS = ["VER", "HAM"]

# Teammate drivers — within-team style control group
# VER vs PER (same Red Bull), HAM vs BOT (same Mercedes)
TEAMMATE_PAIRS = {
    "VER": "PER",
    "HAM": "BOT",
}
ALL_DRIVERS = ["VER", "HAM", "PER", "BOT"]

SESSION_TYPE = "R"

# ─────────────────────────────────────────
# CAR UPGRADE FLAGS
# Cumulative — once introduced, carried forward for rest of season
# Source: publicly documented 2021 F1 upgrade calendar
#
# Red Bull RB16B:
#   R7  France     — major downforce package, start of VER dominant run
#   R10 Britain    — aero refinement
#   R22 AbuDhabi   — final season specification
#
# Mercedes W12:
#   R4  Spain      — significant floor and rear wing package (HAM won)
#   R13 Netherlands — updated floor
#   R18 Brazil     — new ICE token, substantial power gain
#   R22 AbuDhabi   — final season specification
#
# Encoding: 1 = upgrade introduced at this round or earlier, 0 = pre-upgrade
# ─────────────────────────────────────────
# Upgrade rounds as ordered lists — each adds one development level
VER_UPGRADE_ROUNDS = [7, 10, 22]   # ordered list not set
HAM_UPGRADE_ROUNDS = [4, 13, 18, 22]

def get_ver_upgrade_level(round_number):
    """Cumulative upgrade count for Red Bull by this round. Range: 0-3."""
    return sum(1 for r in VER_UPGRADE_ROUNDS if round_number >= r)

def get_ham_upgrade_level(round_number):
    """Cumulative upgrade count for Mercedes by this round. Range: 0-4."""
    return sum(1 for r in HAM_UPGRADE_ROUNDS if round_number >= r)

def get_upgrade_delta(round_number):
    """
    VER upgrade level minus HAM upgrade level.
    Positive = Red Bull ahead in development.
    Negative = Mercedes ahead.
    """
    return get_ver_upgrade_level(round_number) - get_ham_upgrade_level(round_number)


# ─────────────────────────────────────────
# TELEMETRY COLUMN NAMES
# ─────────────────────────────────────────
TEL_SPEED    = "Speed"
TEL_THROTTLE = "Throttle"
TEL_BRAKE    = "Brake"
TEL_GEAR     = "nGear"
TEL_RPM      = "RPM"
TEL_DISTANCE = "Distance"

# ─────────────────────────────────────────
# LAP COLUMN NAMES
# ─────────────────────────────────────────
LAP_TIME_COL     = "LapTime"
LAP_NUMBER_COL   = "LapNumber"
COMPOUND_COL     = "Compound"
TYRE_LIFE_COL    = "TyreLife"
TRACK_STATUS_COL = "TrackStatus"
PIT_OUT_COL      = "PitOutTime"
PIT_IN_COL       = "PitInTime"

# ─────────────────────────────────────────
# FEATURE ENGINEERING CONSTANTS
# ─────────────────────────────────────────
THROTTLE_OFF_THRESHOLD   = 5
FULL_THROTTLE_THRESHOLD  = 98
MIN_BRAKE_ZONE_LENGTH_M  = 20

# Expected stint lengths per compound (laps)
# Used for stint_phase normalisation
# Based on 2021 typical strategy windows
STINT_LENGTH_MAP = {
    "SOFT"  : 15,
    "MEDIUM": 25,
    "HARD"  : 40,
}

# ─────────────────────────────────────────
# SPLIT CONFIGURATION
# Phase 2 uses leave-one-race-out
# TEST_RACE is the held-out race — model never sees it during training
# Chosen as Abu Dhabi: championship decider, maximum strategic significance,
# and the race with the most divergent driving styles in Phase 1 EDA
# ─────────────────────────────────────────
TEST_RACE      = "AbuDhabi"   # held out entirely — never seen during training
TEST_ROUND     = 22

# Val races — used for hyperparameter selection and early stopping
# Chosen as Brazil and Qatar — late season, post-upgrade, different track types
VAL_RACES      = ["Brazil", "Qatar"]
VAL_ROUNDS     = {19, 20}

# All other races form the training set
# Note: Monaco excluded from training due to unique circuit characteristics
# that would add noise rather than signal (street circuit, no meaningful
# lap-time delta comparison due to traffic and safety car periods)
EXCLUDE_ROUNDS = {5}   # Monaco

# ─────────────────────────────────────────
# TARGET COLUMNS
# ─────────────────────────────────────────
TARGET_REGRESSION     = "lap_time_delta_sec"
TARGET_CLASSIFICATION = "ver_faster"

# ─────────────────────────────────────────
# HYPERPARAMETERS (empty — populated after EDA)
# ─────────────────────────────────────────
INITIAL_MODEL_PARAMS = {}

# ─────────────────────────────────────────
# BEST PARAMS (auto-populated by model_trainer.py)
# ─────────────────────────────────────────
BEST_CLASSIFIER_PARAMS = {}
BEST_REGRESSOR_PARAMS  = {}


if __name__ == "__main__":
    from src.logger import logging as log


    log.info("Phase 2 config loaded.")
    log.info(f"Season       : {SEASON}")
    log.info(f"Total races  : {len(RACES)}")
    log.info(f"Drivers      : {DRIVERS}")
    log.info(f"All drivers  : {ALL_DRIVERS}")
    log.info(f"Test race    : {TEST_RACE} (Round {TEST_ROUND})")
    log.info(f"Val races    : {VAL_RACES}")
    log.info(f"Excluded     : Monaco (Round 5) — street circuit noise")
    log.info(f"Training races: {len(RACES) - 1 - len(VAL_RACES) - len(EXCLUDE_ROUNDS)}"
             f" rounds")

    log.info("\nUpgrade levels by round:")
    for rnd, label in RACES:
        ver_f = get_ver_upgrade_level(rnd)
        ham_f = get_ham_upgrade_level(rnd)
        delta = get_upgrade_delta(rnd)
        log.info(f"  R{rnd:>2} {label:<15} VER_level={ver_f} HAM_level={ham_f} "
                 f"delta={delta:+d}")