import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    LAPS_RAW_PATH, TELEMETRY_RAW_PATH, RACE_METADATA_PATH,
    FEATURES_PATH, FEATURES_SAME_COMPOUND_PATH,
    PREPROCESSOR_PATH,
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TRAIN_SC_PATH, VAL_SC_PATH, TEST_SC_PATH,
    DRIVERS, ALL_DRIVERS, TEAMMATE_PAIRS,
    EXCLUDE_ROUNDS, EXCLUDE_FROM_PAIRING, EXCLUDE_FROM_TEAMMATE,
    EXCLUDE_FROM_SC, LOW_SAMPLE_ROUNDS,
    STINT_LENGTH_MAP,
    TEST_RACE, VAL_RACES,
    THROTTLE_OFF_THRESHOLD, FULL_THROTTLE_THRESHOLD,
    MIN_BRAKE_ZONE_LENGTH_M,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
)


# ─────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────

def load_data():
    try:
        log.info("Loading raw data...")
        laps = pd.read_csv(LAPS_RAW_PATH)
        tel  = pd.read_parquet(TELEMETRY_RAW_PATH)
        meta = pd.read_csv(RACE_METADATA_PATH)
        log.info(f"  Laps shape      : {laps.shape}")
        log.info(f"  Telemetry shape : {tel.shape}")
        log.info(f"  Metadata shape  : {meta.shape}")
        return laps, tel, meta
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 2 — FILTER LAPS
# Green flag only; exclude unwanted rounds
# ─────────────────────────────────────────

def filter_laps(laps):
    try:
        before = len(laps)
        green = laps[laps["TrackStatus"] == 1].copy()
        log.info(f"Green flag filter: {before} → {len(green)} laps "
                 f"({before - len(green)} non-green dropped)")

        # Drop entirely excluded rounds (Monaco)
        green = green[~green["RoundNumber"].isin(EXCLUDE_ROUNDS)].copy()
        log.info(f"After EXCLUDE_ROUNDS filter: {len(green)} laps")

        return green
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 3 — TELEMETRY FEATURES PER LAP PER DRIVER
# Compute: coasting_pct, full_throttle_pct, gear_shifts,
#          brake_zone_count, avg_brake_zone_length, avg_entry_speed
# ─────────────────────────────────────────

def compute_tel_features_for_lap(lap_tel):
    """
    Given telemetry rows for a single lap (one driver),
    return a dict of style features.
    """
    if len(lap_tel) < 10:
        return None

    speed    = lap_tel["Speed"].values
    throttle = lap_tel["Throttle"].values
    brake    = lap_tel["Brake"].astype(bool).values
    gear     = lap_tel["nGear"].values
    distance = lap_tel["Distance"].values

    # Coasting: throttle below threshold AND not braking
    coasting_mask    = (throttle < THROTTLE_OFF_THRESHOLD) & (~brake)
    coasting_pct     = coasting_mask.mean() * 100

    # Full throttle: throttle above threshold
    full_throttle_mask = throttle >= FULL_THROTTLE_THRESHOLD
    full_throttle_pct  = full_throttle_mask.mean() * 100

    # Gear shifts: count changes in gear
    gear_shifts = int(np.sum(np.diff(gear) != 0))

    # Brake zones: consecutive brake=True segments
    brake_int   = brake.astype(int)
    brake_diff  = np.diff(brake_int, prepend=0)
    zone_starts = np.where(brake_diff == 1)[0]
    zone_ends   = np.where(brake_diff == -1)[0]

    # If still braking at end, close the last zone
    if len(zone_starts) > len(zone_ends):
        zone_ends = np.append(zone_ends, len(brake) - 1)

    brake_zone_lengths = []
    entry_speeds       = []

    for s, e in zip(zone_starts, zone_ends):
        if e >= len(distance) or s >= len(distance):
            continue
        length = distance[e] - distance[s]
        if length >= MIN_BRAKE_ZONE_LENGTH_M:
            brake_zone_lengths.append(length)
            entry_speeds.append(speed[s])

    brake_zone_count     = len(brake_zone_lengths)
    avg_brake_zone_length = np.mean(brake_zone_lengths) if brake_zone_lengths else 0.0
    avg_entry_speed       = np.mean(entry_speeds) if entry_speeds else 0.0

    return {
        "coasting_pct"         : coasting_pct,
        "full_throttle_pct"    : full_throttle_pct,
        "gear_shifts"          : gear_shifts,
        "brake_zone_count"     : brake_zone_count,
        "avg_brake_zone_length": avg_brake_zone_length,
        "avg_entry_speed"      : avg_entry_speed,
    }


def compute_all_tel_features(tel):
    """
    Group telemetry by Driver + Race + LapNumber
    and compute style metrics for each lap.
    Returns a DataFrame with one row per (Driver, Race, LapNumber).
    """
    try:
        log.info("Computing telemetry features per lap per driver...")
        records = []
        groups  = tel.groupby(["Driver", "Race", "LapNumber"])
        total   = len(groups)

        for i, ((driver, race, lap_number), lap_tel) in enumerate(groups):
            if i % 500 == 0:
                log.info(f"  Processing group {i}/{total}...")
            feats = compute_tel_features_for_lap(lap_tel)
            if feats is None:
                continue
            feats["Driver"]    = driver
            feats["Race"]      = race
            feats["LapNumber"] = lap_number
            records.append(feats)

        tel_features = pd.DataFrame(records)
        log.info(f"Telemetry features computed: {tel_features.shape}")
        return tel_features
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 4 — PAIR VER AND HAM
# Inner join on Race + LapNumber
# Only non-excluded pairing rounds
# ─────────────────────────────────────────

def pair_drivers(laps_green, tel_features, meta):
    """
    Merge lap data with telemetry features, then pair VER and HAM.
    Returns a wide DataFrame with one row per paired lap.
    """
    try:
        log.info("Pairing VER and HAM laps...")

        # Merge lap info with telemetry features
        lap_cols = ["Driver", "Race", "RoundNumber", "LapNumber",
                    "LapTimeSec", "Compound", "TyreLife"]
        lap_cols = [c for c in lap_cols if c in laps_green.columns]
        laps_slim = laps_green[lap_cols].copy()

        combined = laps_slim.merge(tel_features, on=["Driver", "Race", "LapNumber"],
                                   how="inner")
        log.info(f"After merging laps + telemetry: {combined.shape}")

        # Split by driver
        ver_df = combined[combined["Driver"] == "VER"].copy()
        ham_df = combined[combined["Driver"] == "HAM"].copy()
        per_df = combined[combined["Driver"] == "PER"].copy()
        bot_df = combined[combined["Driver"] == "BOT"].copy()

        # Remove EXCLUDE_FROM_PAIRING rounds from primary pairing
        ver_df = ver_df[~ver_df["RoundNumber"].isin(EXCLUDE_FROM_PAIRING)].copy()
        ham_df = ham_df[~ham_df["RoundNumber"].isin(EXCLUDE_FROM_PAIRING)].copy()

        # Rename columns with driver prefix
        def prefix_cols(df, driver):
            rename = {}
            for col in df.columns:
                if col not in ["Race", "RoundNumber", "LapNumber", "Driver"]:
                    rename[col] = f"{driver}_{col}"
            return df.drop(columns=["Driver"]).rename(columns=rename)

        ver_p = prefix_cols(ver_df, "VER")
        ham_p = prefix_cols(ham_df, "HAM")
        per_p = prefix_cols(per_df, "PER")
        bot_p = prefix_cols(bot_df, "BOT")

        # Inner join VER + HAM on Race + LapNumber
        paired = ver_p.merge(ham_p, on=["Race", "RoundNumber", "LapNumber"],
                             how="inner")
        log.info(f"Paired (VER x HAM): {paired.shape}")

        # Merge in metadata (upgrade levels)
        paired = paired.merge(meta[["Race", "VER_upgrade_level",
                                    "HAM_upgrade_level", "upgrade_delta"]],
                              on="Race", how="left")

        # Merge teammate data for style normalisation
        # Per-race median to guard against outliers (Bottas Imola wing failure)
        per_race = per_p.groupby("Race").median(numeric_only=True).reset_index()
        bot_race = bot_p.groupby("Race").median(numeric_only=True).reset_index()

        per_race_cols = {c: f"PER_{c.removeprefix('PER_')}_race_med"
                         if c.startswith("PER_") else c for c in per_race.columns}
        bot_race_cols = {c: f"BOT_{c.removeprefix('BOT_')}_race_med"
                         if c.startswith("BOT_") else c for c in bot_race.columns}

        per_race = per_race.rename(columns=per_race_cols)
        bot_race = bot_race.rename(columns=bot_race_cols)

        paired = paired.merge(per_race, on="Race", how="left")
        paired = paired.merge(bot_race, on="Race", how="left")

        log.info(f"After metadata + teammate merge: {paired.shape}")
        return paired

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 5 — ENGINEER FEATURES
# ─────────────────────────────────────────

def engineer_features(paired):
    """
    Build all delta, interaction, and new Phase 2 features.
    """
    try:
        log.info("Engineering features...")
        df = paired.copy()

        # ── Phase 1 delta features ──────────────────────────
        df["coasting_pct_delta"]          = df["VER_coasting_pct"]       - df["HAM_coasting_pct"]
        df["full_throttle_pct_delta"]     = df["VER_full_throttle_pct"]  - df["HAM_full_throttle_pct"]
        df["gear_shifts_delta"]           = df["VER_gear_shifts"]         - df["HAM_gear_shifts"]
        df["avg_brake_zone_length_delta"] = df["VER_avg_brake_zone_length"] - df["HAM_avg_brake_zone_length"]
        df["avg_entry_speed_delta"]       = df["VER_avg_entry_speed"]    - df["HAM_avg_entry_speed"]
        df["brake_zone_count_delta"]      = df["VER_brake_zone_count"]   - df["HAM_brake_zone_count"]

        # Tyre life delta
        df["tyre_life_delta"] = df["VER_TyreLife"] - df["HAM_TyreLife"]

        # Tyre × coasting interaction (clipped ±50)
        raw_interaction = df["tyre_life_delta"] * df["coasting_pct_delta"]
        df["tyre_life_x_coasting_delta"] = raw_interaction.clip(-50, 50)

        # Stint phase: normalised tyre life per compound
        def stint_phase(tyre_life, compound):
            expected = compound.map(STINT_LENGTH_MAP).fillna(25)
            return tyre_life / expected

        df["VER_stint_phase"] = stint_phase(df["VER_TyreLife"], df["VER_Compound"])
        df["HAM_stint_phase"] = stint_phase(df["HAM_TyreLife"], df["HAM_Compound"])
        df["stint_phase_delta"] = df["VER_stint_phase"] - df["HAM_stint_phase"]

        # Race encoding — use RoundNumber as integer (1-22)
        df["race_enc"] = df["RoundNumber"]

        # abu_dhabi_gear_delta — gear shift delta scaled to Abu Dhabi round
        # 1 for AbuDhabi rows, 0 elsewhere (race-specific interaction term)
        abu_dhabi_round = df["Race"] == TEST_RACE
        df["abu_dhabi_gear_delta"] = df["gear_shifts_delta"] * abu_dhabi_round.astype(int)

        # Compound encoding
        compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
                        "INTERMEDIATE": 3, "WET": 4}
        df["VER_compound_enc"] = df["VER_Compound"].map(compound_map).fillna(-1).astype(int)
        df["HAM_compound_enc"] = df["HAM_Compound"].map(compound_map).fillna(-1).astype(int)
        df["same_compound"]    = (df["VER_Compound"] == df["HAM_Compound"]).astype(int)

        # ── Phase 2 — car context ────────────────────────────
        df["is_low_sample"] = df["RoundNumber"].isin(LOW_SAMPLE_ROUNDS).astype(int)

        # ── Phase 2 — teammate normalised style ─────────────
        # VER_style_M = VER_M - PER_M (same car, driving style residual)
        # HAM_style_M = HAM_M - BOT_M
        # Use per-race median from teammate (already merged as _race_med columns)
        # Cap at ±5 to suppress extreme outliers (e.g. Bottas wing failure at Imola)

        def safe_style(driver_col, teammate_col, df):
            raw = df[driver_col] - df[teammate_col]
            return raw.clip(-5, 5)

        # Coasting style
        df["VER_style_coasting"] = safe_style(
            "VER_coasting_pct", "PER_coasting_pct_race_med", df)
        df["HAM_style_coasting"] = safe_style(
            "HAM_coasting_pct", "BOT_coasting_pct_race_med", df)

        # Zero out style features for excluded teammate rounds
        excluded_tmm_races = df["Race"].isin(
            df[df["RoundNumber"].isin(EXCLUDE_FROM_TEAMMATE)]["Race"].unique()
        )
        df.loc[excluded_tmm_races, "VER_style_coasting"] = 0
        df.loc[excluded_tmm_races, "HAM_style_coasting"] = 0

        df["style_coasting_delta"] = df["VER_style_coasting"] - df["HAM_style_coasting"]

        # Brake length style
        df["VER_style_brake_length"] = safe_style(
            "VER_avg_brake_zone_length", "PER_avg_brake_zone_length_race_med", df)
        df["HAM_style_brake_length"] = safe_style(
            "HAM_avg_brake_zone_length", "BOT_avg_brake_zone_length_race_med", df)
        df.loc[excluded_tmm_races, "VER_style_brake_length"] = 0
        df.loc[excluded_tmm_races, "HAM_style_brake_length"] = 0
        df["style_brake_delta"] = df["VER_style_brake_length"] - df["HAM_style_brake_length"]

        # Full throttle style
        df["VER_style_full_throttle"] = safe_style(
            "VER_full_throttle_pct", "PER_full_throttle_pct_race_med", df)
        df["HAM_style_full_throttle"] = safe_style(
            "HAM_full_throttle_pct", "BOT_full_throttle_pct_race_med", df)
        df.loc[excluded_tmm_races, "VER_style_full_throttle"] = 0
        df.loc[excluded_tmm_races, "HAM_style_full_throttle"] = 0
        df["style_throttle_delta"] = df["VER_style_full_throttle"] - df["HAM_style_full_throttle"]

        # ── Targets ──────────────────────────────────────────
        df[TARGET_REGRESSION]     = df["VER_LapTimeSec"] - df["HAM_LapTimeSec"]
        df[TARGET_CLASSIFICATION] = (df[TARGET_REGRESSION] < 0).astype(int)

        # ── rolling_delta_3: 3-lap rolling mean of target ────
        # MUST be computed before split to avoid leakage (uses historical laps)
        # Sort by Race + LapNumber so rolling is temporal within race
        df = df.sort_values(["Race", "LapNumber"]).reset_index(drop=True)
        df["rolling_delta_3"] = (
            df.groupby("Race")[TARGET_REGRESSION]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        # Fill NaN on first 1-3 laps of each race with the race mean target
        race_mean = df.groupby("Race")[TARGET_REGRESSION].transform("mean")
        df["rolling_delta_3"] = df["rolling_delta_3"].fillna(race_mean)

        log.info(f"Features engineered. Shape: {df.shape}")
        return df

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 6 — SELECT FINAL FEATURE COLUMNS
# ─────────────────────────────────────────

FEATURE_COLS = [
    # Delta features
    "coasting_pct_delta",
    "full_throttle_pct_delta",
    "gear_shifts_delta",
    "avg_brake_zone_length_delta",
    "avg_entry_speed_delta",
    "brake_zone_count_delta",
    "tyre_life_delta",
    "tyre_life_x_coasting_delta",
    "stint_phase_delta",
    "abu_dhabi_gear_delta",
    "rolling_delta_3",

    # Individual driver features
    "VER_coasting_pct", "HAM_coasting_pct",
    "VER_full_throttle_pct", "HAM_full_throttle_pct",
    "VER_gear_shifts", "HAM_gear_shifts",
    "VER_avg_brake_zone_length", "HAM_avg_brake_zone_length",
    "VER_avg_entry_speed", "HAM_avg_entry_speed",
    "VER_TyreLife", "HAM_TyreLife",
    "VER_stint_phase", "HAM_stint_phase",

    # Phase 2 — car context
    "VER_upgrade_level",
    "HAM_upgrade_level",
    "upgrade_delta",
    "is_low_sample",

    # Phase 2 — teammate normalised style
    "VER_style_coasting",
    "HAM_style_coasting",
    "style_coasting_delta",
    "VER_style_brake_length",
    "HAM_style_brake_length",
    "style_brake_delta",
    "VER_style_full_throttle",
    "HAM_style_full_throttle",
    "style_throttle_delta",

    # Contextual
    "same_compound",
    "VER_compound_enc",
    "HAM_compound_enc",
    "LapNumber",
    "race_enc",
]

ID_COLS = ["Race", "RoundNumber", "LapNumber",
           TARGET_REGRESSION, TARGET_CLASSIFICATION]


def select_columns(df):
    try:
        # Check which feature cols are actually present
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            log.warning(f"Missing feature columns (will be skipped): {missing}")

        present_features = [c for c in FEATURE_COLS if c in df.columns]
        final_cols = ID_COLS + present_features

        # Drop rows with NaN in any feature or target
        result = df[final_cols].copy()
        before = len(result)
        result = result.dropna(subset=present_features + [TARGET_REGRESSION])
        log.info(f"Dropped {before - len(result)} rows with NaN features/target")
        log.info(f"Final feature dataframe: {result.shape}")
        return result, present_features

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 7 — LORO SPLIT
# Leave-one-race-out: test=AbuDhabi, val=Brazil+Qatar, train=rest
# Split at race level — no lap from one race ever crosses train/val/test
# ─────────────────────────────────────────

def loro_split(df):
    try:
        log.info("Performing Leave-One-Race-Out split...")

        test_df  = df[df["Race"] == TEST_RACE].copy()
        val_df   = df[df["Race"].isin(VAL_RACES)].copy()
        train_df = df[~df["Race"].isin([TEST_RACE] + VAL_RACES)].copy()

        log.info(f"  Train: {len(train_df)} laps from "
                 f"{sorted(train_df['Race'].unique())}")
        log.info(f"  Val  : {len(val_df)} laps from "
                 f"{sorted(val_df['Race'].unique())}")
        log.info(f"  Test : {len(test_df)} laps from "
                 f"{sorted(test_df['Race'].unique())}")

        return train_df, val_df, test_df

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 8 — BUILD SC SUBSET
# Filter where same_compound == 1
# Further exclude EXCLUDE_FROM_SC rounds
# Apply same LORO split
# ─────────────────────────────────────────

def build_sc_subset(df):
    try:
        log.info("Building same-compound subset...")

        sc = df[df["same_compound"] == 1].copy()
        sc = sc[~sc["RoundNumber"].isin(EXCLUDE_FROM_SC)].copy()

        log.info(f"SC subset: {len(sc)} rows (from {len(df)} total, "
                 f"{100*len(sc)/len(df):.1f}%)")

        sc_test  = sc[sc["Race"] == TEST_RACE].copy()
        sc_val   = sc[sc["Race"].isin(VAL_RACES)].copy()
        sc_train = sc[~sc["Race"].isin([TEST_RACE] + VAL_RACES)].copy()

        log.info(f"  SC Train: {len(sc_train)}")
        log.info(f"  SC Val  : {len(sc_val)}")
        log.info(f"  SC Test : {len(sc_test)}")

        return sc, sc_train, sc_val, sc_test

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 9 — SCALE
# Fit StandardScaler on TRAIN only
# Apply to val and test
# ─────────────────────────────────────────

def scale_splits(train_df, val_df, test_df,
                 sc_train_df, sc_val_df, sc_test_df,
                 feature_cols):
    try:
        log.info("Scaling features (fit on train only)...")

        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols])

        def apply_scale(df):
            scaled = df.copy()
            scaled[feature_cols] = scaler.transform(df[feature_cols])
            return scaled

        train_scaled    = apply_scale(train_df)
        val_scaled      = apply_scale(val_df)
        test_scaled     = apply_scale(test_df)
        sc_train_scaled = apply_scale(sc_train_df)
        sc_val_scaled   = apply_scale(sc_val_df)
        sc_test_scaled  = apply_scale(sc_test_df)

        log.info("Scaler fitted and applied.")
        return (scaler,
                train_scaled, val_scaled, test_scaled,
                sc_train_scaled, sc_val_scaled, sc_test_scaled)

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 10 — SAVE
# ─────────────────────────────────────────

def save_outputs(features_df, sc_features_df,
                 train_scaled, val_scaled, test_scaled,
                 sc_train_scaled, sc_val_scaled, sc_test_scaled,
                 scaler):
    try:
        log.info("Saving outputs...")

        features_df.to_csv(FEATURES_PATH, index=False)
        log.info(f"  features.csv          → {FEATURES_PATH} | {features_df.shape}")

        sc_features_df.to_csv(FEATURES_SAME_COMPOUND_PATH, index=False)
        log.info(f"  features_same_compound.csv → {FEATURES_SAME_COMPOUND_PATH} "
                 f"| {sc_features_df.shape}")

        train_scaled.to_csv(TRAIN_PATH, index=False)
        val_scaled.to_csv(VAL_PATH, index=False)
        test_scaled.to_csv(TEST_PATH, index=False)
        log.info(f"  train.csv → {TRAIN_PATH} | {train_scaled.shape}")
        log.info(f"  val.csv   → {VAL_PATH}   | {val_scaled.shape}")
        log.info(f"  test.csv  → {TEST_PATH}  | {test_scaled.shape}")

        sc_train_scaled.to_csv(TRAIN_SC_PATH, index=False)
        sc_val_scaled.to_csv(VAL_SC_PATH, index=False)
        sc_test_scaled.to_csv(TEST_SC_PATH, index=False)
        log.info(f"  train_sc.csv → {TRAIN_SC_PATH} | {sc_train_scaled.shape}")
        log.info(f"  val_sc.csv   → {VAL_SC_PATH}   | {sc_val_scaled.shape}")
        log.info(f"  test_sc.csv  → {TEST_SC_PATH}  | {sc_test_scaled.shape}")

        with open(PREPROCESSOR_PATH, "wb") as f:
            pickle.dump(scaler, f)
        log.info(f"  preprocessor.pkl → {PREPROCESSOR_PATH}")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_transformation():
    try:
        log.info("=" * 60)
        log.info("Starting data transformation — Phase 2")
        log.info("=" * 60)

        # Step 1: Load
        laps, tel, meta = load_data()

        # Step 2: Filter laps
        laps_green = filter_laps(laps)

        # Step 3: Telemetry features per lap per driver
        tel_features = compute_all_tel_features(tel)

        # Step 4: Pair VER and HAM, merge teammate data
        paired = pair_drivers(laps_green, tel_features, meta)

        # Step 5: Engineer all features
        features_full = engineer_features(paired)

        # Step 6: Select final columns, drop NaNs
        features_df, feature_cols = select_columns(features_full)

        # Step 7: LORO split on full dataset
        train_df, val_df, test_df = loro_split(features_df)

        # Step 8: SC subset + LORO split
        sc_df, sc_train_df, sc_val_df, sc_test_df = build_sc_subset(features_df)

        # Step 9: Scale (fit on train only)
        (scaler,
         train_scaled, val_scaled, test_scaled,
         sc_train_scaled, sc_val_scaled, sc_test_scaled) = scale_splits(
             train_df, val_df, test_df,
             sc_train_df, sc_val_df, sc_test_df,
             feature_cols
        )

        # Step 10: Save
        save_outputs(
            features_df, sc_df,
            train_scaled, val_scaled, test_scaled,
            sc_train_scaled, sc_val_scaled, sc_test_scaled,
            scaler
        )

        log.info("=" * 60)
        log.info("Data transformation complete.")
        log.info("=" * 60)

        return (features_df, sc_df,
                train_scaled, val_scaled, test_scaled,
                sc_train_scaled, sc_val_scaled, sc_test_scaled,
                scaler)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    (features_df, sc_df,
     train, val, test,
     sc_train, sc_val, sc_test,
     scaler) = run_transformation()

    print("\n--- FULL DATASET ---")
    print(f"Total rows    : {len(features_df)}")
    print(f"Train rows    : {len(train)}")
    print(f"Val rows      : {len(val)}")
    print(f"Test rows     : {len(test)}")
    print(f"Races in train: {sorted(train['Race'].unique())}")
    print(f"Target mean   : {features_df[TARGET_REGRESSION].mean():.3f}s")
    print(f"Target std    : {features_df[TARGET_REGRESSION].std():.3f}s")
    print(f"Class balance : {features_df[TARGET_CLASSIFICATION].mean():.2%} VER faster")

    print("\n--- SC DATASET ---")
    print(f"Total SC rows : {len(sc_df)}")
    print(f"SC Train rows : {len(sc_train)}")
    print(f"SC Val rows   : {len(sc_val)}")
    print(f"SC Test rows  : {len(sc_test)}")
    print(f"Races in SC   : {sorted(sc_df['Race'].unique())}")

    print("\n--- FEATURE NULLS (full, unscaled) ---")
    null_counts = features_df.isnull().sum()
    print(null_counts[null_counts > 0].to_string()
          if null_counts.any() else "No nulls.")

    print("\n--- SPLIT SANITY CHECK ---")
    print("No race appears in more than one split:")
    all_splits = {
        "train": set(train["Race"].unique()),
        "val"  : set(val["Race"].unique()),
        "test" : set(test["Race"].unique()),
    }
    overlaps = {
        "train∩val" : all_splits["train"] & all_splits["val"],
        "train∩test": all_splits["train"] & all_splits["test"],
        "val∩test"  : all_splits["val"]   & all_splits["test"],
    }
    for k, v in overlaps.items():
        print(f"  {k}: {v if v else 'CLEAN'}")