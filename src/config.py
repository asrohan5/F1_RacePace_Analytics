import os

BASE_DIR = os.getcwd()

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SPLITS_DIR,
             MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Raw data
LAPS_RAW_PATH = os.path.join(DATA_RAW_DIR, "laps_raw.csv")
TELEMETRY_RAW_PATH = os.path.join(DATA_RAW_DIR, "telemetry_raw.parquet")
RACE_METADATA_PATH = os.path.join(DATA_RAW_DIR, "race_metadata.csv")

# Processed features
FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "features.csv")
FEATURES_SAME_COMPOUND_PATH = os.path.join(DATA_PROCESSED_DIR, "features_same_compound.csv")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

# Train/val/test splits
TRAIN_PATH = os.path.join(DATA_SPLITS_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_SPLITS_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_SPLITS_DIR, "test.csv")
TRAIN_SC_PATH = os.path.join(DATA_SPLITS_DIR, "train_sc.csv")
VAL_SC_PATH = os.path.join(DATA_SPLITS_DIR, "val_sc.csv")
TEST_SC_PATH = os.path.join(DATA_SPLITS_DIR, "test_sc.csv")

# Saved models
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
REGRESSOR_PATH = os.path.join(MODELS_DIR, "regressor.pkl")
CLASSIFIER_SC_PATH = os.path.join(MODELS_DIR, "classifier_sc.pkl")
REGRESSOR_SC_PATH = os.path.join(MODELS_DIR, "regressor_sc.pkl")

# CV fold scores saved by model_trainer for use in diagnostics
CV_SCORES_PATH = os.path.join(ARTIFACTS_DIR, "cv_scores.json")

SEASON = 2021

RACES = [
    (1, "Bahrain"),
    (2, "Imola"),
    (3, "Portugal"),
    (4, "Spain"),
    (5, "Monaco"),
    (6, "Azerbaijan"),
    (7, "France"),
    (8, "Styria"),
    (9, "Austria"),
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

DRIVERS = ["VER", "HAM"]

# Within-team pairs used for teammate-normalised style features.
# VER minus PER isolates driver style from Red Bull car pace.
# HAM minus BOT isolates driver style from Mercedes car pace.
TEAMMATE_PAIRS = {
    "VER": "PER",
    "HAM": "BOT",
}
ALL_DRIVERS = ["VER", "HAM", "PER", "BOT"]

SESSION_TYPE = "R"

# Car upgrade encoding — cumulative development levels per team.
# Each entry marks the round at which a meaningful aero or power unit
# upgrade was introduced. Once introduced the level carries forward
# for the rest of the season.
#
# Red Bull RB16B:
#   R7  France      major downforce package, start of VER dominant run
#   R10 Britain     aero refinement
#   R22 AbuDhabi    final season specification
#
# Mercedes W12:
#   R4  Spain       floor and rear wing package
#   R13 Netherlands updated floor
#   R18 Brazil      new ICE token, substantial power unit gain
#   R22 AbuDhabi    final season specification
VER_UPGRADE_ROUNDS = [7, 10, 22]
HAM_UPGRADE_ROUNDS = [4, 13, 18, 22]


def get_ver_upgrade_level(round_number):
    """Cumulative Red Bull upgrade count by this round. Range 0-3."""
    return sum(1 for r in VER_UPGRADE_ROUNDS if round_number >= r)


def get_ham_upgrade_level(round_number):
    """Cumulative Mercedes upgrade count by this round. Range 0-4."""
    return sum(1 for r in HAM_UPGRADE_ROUNDS if round_number >= r)


def get_upgrade_delta(round_number):
    """
    VER upgrade level minus HAM upgrade level.
    Positive means Red Bull ahead in development, negative means Mercedes ahead.
    """
    return get_ver_upgrade_level(round_number) - get_ham_upgrade_level(round_number)


# Telemetry column names
TEL_SPEED = "Speed"
TEL_THROTTLE = "Throttle"
TEL_BRAKE = "Brake"
TEL_GEAR = "nGear"
TEL_RPM = "RPM"
TEL_DISTANCE = "Distance"

# Lap data column names
LAP_TIME_COL = "LapTime"
LAP_NUMBER_COL = "LapNumber"
COMPOUND_COL = "Compound"
TYRE_LIFE_COL = "TyreLife"
TRACK_STATUS_COL = "TrackStatus"
PIT_OUT_COL = "PitOutTime"
PIT_IN_COL = "PitInTime"

# Telemetry thresholds
THROTTLE_OFF_THRESHOLD = 5
FULL_THROTTLE_THRESHOLD = 98
MIN_BRAKE_ZONE_LENGTH_M = 20

# Expected stint lengths per compound in laps, based on 2021 typical strategy windows.
# Used to normalise tyre life into a 0-1 stint phase value.
STINT_LENGTH_MAP = {
    "SOFT": 15,
    "MEDIUM": 25,
    "HARD": 40,
}

# Track overtake difficulty — static per-circuit encoding.
# Captures how much race position and DRS effects dominate the lap time delta
# relative to pure driving style. High values indicate circuits where position
# effects are strong and the style signal is more confounded.
# 1 = low (tight circuits, limited overtaking)
# 2 = medium (most mixed-character circuits)
# 3 = high (long straights, strong DRS effect, position changes frequently)
TRACK_OVERTAKE_DIFFICULTY = {
    "Bahrain": 2,
    "Imola": 2,
    "Portugal": 2,
    "Spain": 2,
    "Monaco": 1,
    "Azerbaijan": 3,
    "France": 2,
    "Styria": 2,
    "Austria": 2,
    "Britain": 2,
    "Hungary": 1,
    "Belgium": 3,
    "Netherlands": 1,
    "Italy": 3,
    "Russia": 2,
    "Turkey": 2,
    "UnitedStates": 2,
    "Mexico": 2,
    "Brazil": 3,
    "Qatar": 2,
    "SaudiArabia": 3,
    "AbuDhabi": 2,
}

# Split configuration
# Abu Dhabi chosen as test race: championship decider, maximum strategic
# significance, and the race with the most divergent driving styles in EDA.
TEST_RACE = "AbuDhabi"
TEST_ROUND = 22

# Val races are an adversarial choice, not a random sample.
# Brazil (R19): Hamilton won from P5 after Verstappen front wing penalty.
#   HAM pace advantage was driven by race position and strategy, not style.
#   Tests whether the classifier over-predicts VER on HAM-dominant laps.
# Qatar (R20): Verstappen grid penalty. HAM dominant throughout.
#   Lowest expected classifier accuracy of any race in the season.
#   Chosen as a worst-case generalisation test rather than a representative hold-out.
VAL_RACES = ["Brazil", "Qatar"]
VAL_ROUNDS = {19, 20}

# Monaco excluded entirely — street circuit with no meaningful lap-time delta
# comparison between the two drivers due to traffic and safety car periods.
EXCLUDE_ROUNDS = {5}

# Rounds excluded from VER vs HAM paired analysis.
# One or both drivers did not complete representative racing laps.
# Britain (R10): VER retired on lap 1 after collision with HAM.
# Hungary (R11): VER rejoined at the back after lap 1 incident, laps unrepresentative.
# Belgium (R12): race behind safety car for most of the distance, no real racing.
# Russia (R15): VER grid penalty and strategic pit stop, delta not meaningful.
EXCLUDE_FROM_PAIRING = {10, 11, 12, 15}

# Rounds where teammate normalisation is unreliable.
# Hungary (R11): both Bottas and Perez retired early, insufficient laps for race median.
# SaudiArabia (R21): Perez completed only 9 laps before retirement.
EXCLUDE_FROM_TEAMMATE = {11, 21}

# Rounds excluded from the same-compound subset model.
# Italy (R14): 0% of paired laps were same-compound.
# Russia (R15): 0% same-compound.
# SaudiArabia (R21): 28% same-compound, below the minimum threshold for reliable training.
EXCLUDE_FROM_SC = {14, 15, 21}

# Rounds included in training but flagged as low reliability.
# Results from these races should be interpreted cautiously.
# Imola (R2): sprint format weekend, reduced green-flag lap count, Bottas incident.
# Italy (R14): multiple incidents, very few clean paired laps.
# SaudiArabia (R21): chaotic race with multiple restarts and safety car periods.
LOW_SAMPLE_ROUNDS = {2, 14, 21}

# Targets
TARGET_REGRESSION = "lap_time_delta_sec"
TARGET_CLASSIFICATION = "ver_faster"

INITIAL_MODEL_PARAMS = {
    "regressor": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 8,
        "random_state": 42
    },
    "classifier": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42
    }
}

BEST_CLASSIFIER_PARAMS = {}
BEST_REGRESSOR_PARAMS = {}


if __name__ == "__main__":
    from src.logger import logging as log

    log.info(f"Season: {SEASON}")
    log.info(f"Total races: {len(RACES)}")
    log.info(f"Drivers: {DRIVERS}")
    log.info(f"Test race: {TEST_RACE} (Round {TEST_ROUND})")
    log.info(f"Val races: {VAL_RACES}")

    log.info("Upgrade levels by round:")
    for rnd, label in RACES:
        ver_f = get_ver_upgrade_level(rnd)
        ham_f = get_ham_upgrade_level(rnd)
        delta = get_upgrade_delta(rnd)
        log.info(f"  R{rnd:>2} {label:<15} VER={ver_f} HAM={ham_f} delta={delta:+d}")

    log.info("Exclusion summary:")
    log.info(f"  Excluded entirely: {EXCLUDE_ROUNDS}")
    log.info(f"  Excluded from pairing: {EXCLUDE_FROM_PAIRING}")
    log.info(f"  Excluded from SC model: {EXCLUDE_FROM_SC}")
    log.info(f"  Excluded from teammate normalisation: {EXCLUDE_FROM_TEAMMATE}")
    log.info(f"  Low sample rounds: {LOW_SAMPLE_ROUNDS}")

    log.info("Race split assignments:")
    for rnd, label in RACES:
        if rnd in EXCLUDE_ROUNDS or rnd in EXCLUDE_FROM_PAIRING:
            status = "EXCLUDED"
        elif rnd == TEST_ROUND:
            status = "TEST"
        elif label in VAL_RACES:
            status = "VAL"
        elif rnd in LOW_SAMPLE_ROUNDS:
            status = "TRAIN (low sample)"
        else:
            status = "TRAIN"
        log.info(f"  R{rnd:>2} {label:<15} {status}")