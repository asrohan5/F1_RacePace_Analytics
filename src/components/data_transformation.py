import os
import sys
import pandas as pd
import numpy as np
import pickle

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    LAPS_RAW_PATH, TELEMETRY_RAW_PATH,
    FEATURES_PATH, FEATURES_SAME_COMPOUND_PATH,
    PREPROCESSOR_PATH,
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TRAIN_SC_PATH, VAL_SC_PATH, TEST_SC_PATH,
    MIN_BRAKE_ZONE_LENGTH_M,
    THROTTLE_OFF_THRESHOLD, FULL_THROTTLE_THRESHOLD,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
    DRIVERS
)



# ─────────────────────────────────────────
# STEP 1 — LOAD RAW DATA
# ─────────────────────────────────────────

def load_raw_data():
    try:
        laps = pd.read_csv(LAPS_RAW_PATH)
        tel  = pd.read_parquet(TELEMETRY_RAW_PATH)
        log.info(f"Laps loaded      : {laps.shape}")
        log.info(f"Telemetry loaded : {tel.shape}")
        return laps, tel
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 2 — FILTER LAPS
# ─────────────────────────────────────────

def filter_laps(laps):
    """
    Keep only green-flag laps (TrackStatus == 1).
    FastF1 uses compound status codes — '1' is pure green flag.
    """
    try:
        before = len(laps)
        laps = laps[laps["TrackStatus"] == 1].copy()
        after = len(laps)
        log.info(f"Track status filter: {before} → {after} laps "
                 f"(dropped {before - after} non-green laps)")
        return laps
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 3 — ENGINEER TELEMETRY FEATURES PER LAP
# ─────────────────────────────────────────

def compute_telemetry_features(tel):
    """
    For each (Race, Driver, LapNumber), compute driving style features
    from raw telemetry signals.
    Returns one row per lap.
    """
    try:
        records = []

        groups = tel.groupby(["Race", "Driver", "LapNumber"])
        total  = len(groups)
        log.info(f"Computing telemetry features for {total} lap groups...")

        for (race, driver, lap_num), lap_tel in groups:

            lap_tel = lap_tel.sort_values("Distance").reset_index(drop=True)
            brake   = lap_tel["Brake"].astype(bool)
            throttle = lap_tel["Throttle"]
            speed    = lap_tel["Speed"]
            gear     = lap_tel["nGear"]
            distance = lap_tel["Distance"]

            # ── Brake zone detection ──
            # Label each sample with a zone id (changes every time Brake flips)
            zone_id = (brake != brake.shift()).cumsum()

            # Get length of each brake zone in metres
            brake_zone_lengths = (
                lap_tel[brake]
                .groupby(zone_id[brake])["Distance"]
                .agg(lambda x: x.max() - x.min())
            )

            # Keep only real zones (above noise threshold)
            real_zones = brake_zone_lengths[
                brake_zone_lengths >= MIN_BRAKE_ZONE_LENGTH_M
            ]

            brake_zone_count      = len(real_zones)
            avg_brake_zone_length = real_zones.mean() if brake_zone_count > 0 else 0.0

            # ── Entry speed at each real brake zone start ──
            real_zone_ids = real_zones.index
            brake_start_idx = lap_tel[
                brake & (brake != brake.shift()) &
                zone_id.isin(real_zone_ids)
            ].index
            entry_speeds = speed.loc[brake_start_idx]
            avg_entry_speed = entry_speeds.mean() if len(entry_speeds) > 0 else speed.mean()

            # ── Coasting % (neither throttle nor brake) ──
            coasting = ((throttle < THROTTLE_OFF_THRESHOLD) & (~brake))
            coasting_pct = coasting.mean() * 100

            # ── Full throttle % ──
            full_throttle_pct = (throttle >= FULL_THROTTLE_THRESHOLD).mean() * 100

            # ── Gear shifts ──
            gear_shifts = (gear.diff().abs() > 0).sum()

            records.append({
                "Race"                 : race,
                "Driver"               : driver,
                "LapNumber"            : lap_num,
                "brake_zone_count"     : brake_zone_count,
                "avg_brake_zone_length": round(avg_brake_zone_length, 2),
                "avg_entry_speed"      : round(avg_entry_speed, 2),
                "coasting_pct"         : round(coasting_pct, 4),
                "full_throttle_pct"    : round(full_throttle_pct, 4),
                "gear_shifts"          : int(gear_shifts),
            })

        tel_features = pd.DataFrame(records)
        log.info(f"Telemetry features computed: {tel_features.shape}")
        return tel_features

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 4 — PAIR VER AND HAM ON SAME LAP
# ─────────────────────────────────────────

def pair_drivers(laps, tel_features):
    """
    Inner join VER and HAM on Race + LapNumber.
    Only laps where both drivers completed a clean, green-flag lap are kept.
    Each row in the output = one matched lap (one race lap, both drivers).
    """
    try:
        # Split laps by driver
        ver_laps = laps[laps["Driver"] == "VER"].copy()
        ham_laps = laps[laps["Driver"] == "HAM"].copy()

        # Split telemetry features by driver
        ver_tel = tel_features[tel_features["Driver"] == "VER"].copy()
        ham_tel = tel_features[tel_features["Driver"] == "HAM"].copy()

        # Rename columns with driver prefix
        lap_cols     = ["LapTimeSec", "Compound", "TyreLife"]
        tel_cols     = ["brake_zone_count", "avg_brake_zone_length",
                        "avg_entry_speed", "coasting_pct",
                        "full_throttle_pct", "gear_shifts"]

        for col in lap_cols:
            ver_laps = ver_laps.rename(columns={col: f"VER_{col}"})
            ham_laps = ham_laps.rename(columns={col: f"HAM_{col}"})

        for col in tel_cols:
            ver_tel = ver_tel.rename(columns={col: f"VER_{col}"})
            ham_tel = ham_tel.rename(columns={col: f"HAM_{col}"})

        # Merge laps
        paired = ver_laps[["Race", "LapNumber", "RoundNumber",
                            "VER_LapTimeSec", "VER_Compound", "VER_TyreLife"]].merge(
                 ham_laps[["Race", "LapNumber",
                            "HAM_LapTimeSec", "HAM_Compound", "HAM_TyreLife"]],
                 on=["Race", "LapNumber"], how="inner")

        # Merge telemetry features
        paired = paired.merge(
            ver_tel[["Race", "LapNumber"] + [f"VER_{c}" for c in tel_cols]],
            on=["Race", "LapNumber"], how="inner"
        )
        paired = paired.merge(
            ham_tel[["Race", "LapNumber"] + [f"HAM_{c}" for c in tel_cols]],
            on=["Race", "LapNumber"], how="inner"
        )

        log.info(f"Paired laps (inner join): {len(paired)} rows")
        return paired

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 5 — FEATURE ENGINEERING ON PAIRED DATA
# ─────────────────────────────────────────

def engineer_features(paired):
    """
    Build final feature set:
    - Delta features (VER - HAM) for each telemetry signal
    - same_compound flag
    - Compound encoding
    - Lap number (fuel proxy)
    - Race encoding
    - Target variables
    """
    try:
        df = paired.copy()

        # ── Delta features (the primary signal for the model) ──
        df["coasting_pct_delta"]         = df["VER_coasting_pct"]         - df["HAM_coasting_pct"]
        df["full_throttle_pct_delta"]    = df["VER_full_throttle_pct"]    - df["HAM_full_throttle_pct"]
        df["gear_shifts_delta"]          = df["VER_gear_shifts"]          - df["HAM_gear_shifts"]
        df["avg_brake_zone_length_delta"]= df["VER_avg_brake_zone_length"]- df["HAM_avg_brake_zone_length"]
        df["avg_entry_speed_delta"]      = df["VER_avg_entry_speed"]      - df["HAM_avg_entry_speed"]
        df["brake_zone_count_delta"]     = df["VER_brake_zone_count"]     - df["HAM_brake_zone_count"]
        df["tyre_life_delta"]            = df["VER_TyreLife"]             - df["HAM_TyreLife"]
        
        # ── Update after model_diagnostics: Tyre life interaction feature ──
        # Captures whether VER's coasting advantage changes as tyre age difference grows
        # High positive = VER coasts more AND has older tyres (style under pressure)
        # High negative = VER coasts more BUT has fresher tyres (style on new rubber)
        df["tyre_life_x_coasting_delta"] = df["tyre_life_delta"] * df["coasting_pct_delta"]

        # ── same_compound flag ──
        df["same_compound"] = (df["VER_Compound"] == df["HAM_Compound"]).astype(int)

        # ── Compound encoding (VER compound — reference driver) ──
        compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
        df["VER_compound_enc"] = df["VER_Compound"].map(compound_map).fillna(1).astype(int)
        df["HAM_compound_enc"] = df["HAM_Compound"].map(compound_map).fillna(1).astype(int)

        # ── Race encoding ──
        race_map = {"Bahrain": 0, "Spain": 1, "AbuDhabi": 2}
        df["race_enc"] = df["Race"].map(race_map)

        # ── Target variables ──
        df[TARGET_REGRESSION]     = df["VER_LapTimeSec"] - df["HAM_LapTimeSec"]
        df[TARGET_CLASSIFICATION] = (df[TARGET_REGRESSION] < 0).astype(int)

        log.info(f"Features engineered. Final shape: {df.shape}")
        log.info(f"Target regression   — mean={df[TARGET_REGRESSION].mean():.3f}s  "
                 f"std={df[TARGET_REGRESSION].std():.3f}s")
        log.info(f"Target classification — class balance: "
                 f"{df[TARGET_CLASSIFICATION].value_counts().to_dict()}")

        return df

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 6 — SELECT FINAL FEATURE COLUMNS
# ─────────────────────────────────────────

def get_feature_columns():
    """
    Returns the ordered list of feature columns used for modelling.
    Keeping individual VER/HAM features alongside delta features
    gives the model both absolute and relative information.
    """
    features = [
        # Delta features — primary signal
        "coasting_pct_delta",
        "full_throttle_pct_delta",
        "gear_shifts_delta",
        "avg_brake_zone_length_delta",
        "avg_entry_speed_delta",
        "brake_zone_count_delta",
        "tyre_life_delta",
        
        "tyre_life_x_coasting_delta",

        # Individual driver features — context
        "VER_coasting_pct",
        "HAM_coasting_pct",
        "VER_full_throttle_pct",
        "HAM_full_throttle_pct",
        "VER_gear_shifts",
        "HAM_gear_shifts",
        "VER_avg_brake_zone_length",
        "HAM_avg_brake_zone_length",
        "VER_avg_entry_speed",
        "HAM_avg_entry_speed",
        "VER_TyreLife",
        "HAM_TyreLife",

        # Contextual features
        "same_compound",
        "VER_compound_enc",
        "HAM_compound_enc",
        "LapNumber",
        "race_enc",
    ]
    return features


# ─────────────────────────────────────────
# STEP 7 — TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────

def split_data(df):
    """
    Split per race to avoid leakage.
    Within each race, split chronologically (by LapNumber) —
    not randomly, because laps are time-ordered and shuffling
    would leak future lap context into training.
    """
    try:
        train_parts, val_parts, test_parts = [], [], []

        for race in df["Race"].unique():
            race_df = df[df["Race"] == race].sort_values("LapNumber").reset_index(drop=True)
            n       = len(race_df)

            train_end = int(n * TRAIN_RATIO)
            val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

            train_parts.append(race_df.iloc[:train_end])
            val_parts.append(race_df.iloc[train_end:val_end])
            test_parts.append(race_df.iloc[val_end:])

            log.info(f"  {race}: total={n} | train={train_end} | "
                     f"val={val_end - train_end} | test={n - val_end}")

        train_df = pd.concat(train_parts, ignore_index=True)
        val_df   = pd.concat(val_parts,   ignore_index=True)
        test_df  = pd.concat(test_parts,  ignore_index=True)

        log.info(f"Split complete — train={len(train_df)} | "
                 f"val={len(val_df)} | test={len(test_df)}")

        return train_df, val_df, test_df

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# STEP 8 — SCALE FEATURES
# ─────────────────────────────────────────

def scale_features(train_df, val_df, test_df, feature_cols):
    """
    Fit StandardScaler on train only.
    Apply same scaler to val and test.
    Save scaler to disk for use in test_pipeline.
    """
    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        train_scaled = train_df.copy()
        val_scaled   = val_df.copy()
        test_scaled  = test_df.copy()

        train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_scaled[feature_cols]   = scaler.transform(val_df[feature_cols])
        test_scaled[feature_cols]  = scaler.transform(test_df[feature_cols])

        # Save scaler
        with open(PREPROCESSOR_PATH, "wb") as f:
            pickle.dump(scaler, f)

        log.info(f"Scaler fitted on train and saved → {PREPROCESSOR_PATH}")
        log.info(f"Feature means (train): "
                 f"{dict(zip(feature_cols, scaler.mean_.round(4)))}")

        return train_scaled, val_scaled, test_scaled, scaler

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_transformation():
    try:
        log.info("=" * 60)
        log.info("Starting data transformation")
        log.info("=" * 60)

        # Step 1 — Load
        laps, tel = load_raw_data()

        # Step 2 — Filter
        laps = filter_laps(laps)

        # Step 3 — Telemetry features
        tel_features = compute_telemetry_features(tel)

        # Step 4 — Pair drivers
        paired = pair_drivers(laps, tel_features)

        # Step 5 — Feature engineering
        df = engineer_features(paired)

        # Step 6 — Feature columns
        feature_cols = get_feature_columns()
        log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

        # Save full feature set before splitting (useful for EDA follow-up)
        df.to_csv(FEATURES_PATH, index=False)
        log.info(f"Full feature set saved → {FEATURES_PATH}  | shape: {df.shape}")

        # Step 7 — Split
        train_df, val_df, test_df = split_data(df)

        # Step 8 — Scale
        train_df, val_df, test_df, scaler = scale_features(
            train_df, val_df, test_df, feature_cols
        )

        # Save splits
        train_df.to_csv(TRAIN_PATH, index=False)
        val_df.to_csv(VAL_PATH,     index=False)
        test_df.to_csv(TEST_PATH,   index=False)

        log.info(f"Train saved → {TRAIN_PATH}")
        log.info(f"Val   saved → {VAL_PATH}")
        log.info(f"Test  saved → {TEST_PATH}")

        # ── Save same-compound subset ──
        # Primary model will train on this — isolates driving style from tyre strategy
        df_same = df[df["same_compound"] == 1].copy()
        df_same.to_csv(FEATURES_SAME_COMPOUND_PATH, index=False)
        log.info(f"Same-compound subset saved → {FEATURES_SAME_COMPOUND_PATH} "
                f"| shape: {df_same.shape} "
                f"| {len(df_same)} of {len(df)} paired laps ({100*len(df_same)/len(df):.1f}%)")

        # Split same-compound subset
        log.info("Splitting same-compound subset...")
        train_sc, val_sc, test_sc = split_data(df_same)

        # Scale same-compound splits using same scaler fitted on full train
        feature_cols_sc = get_feature_columns()
        train_sc_scaled = train_sc.copy()
        val_sc_scaled   = val_sc.copy()
        test_sc_scaled  = test_sc.copy()

        train_sc_scaled[feature_cols_sc] = scaler.transform(train_sc[feature_cols_sc])
        val_sc_scaled[feature_cols_sc]   = scaler.transform(val_sc[feature_cols_sc])
        test_sc_scaled[feature_cols_sc]  = scaler.transform(test_sc[feature_cols_sc])

        train_sc_scaled.to_csv(TRAIN_SC_PATH, index=False)
        val_sc_scaled.to_csv(VAL_SC_PATH,     index=False)
        test_sc_scaled.to_csv(TEST_SC_PATH,   index=False)

        log.info(f"Same-compound splits — train={len(train_sc)} | "
                f"val={len(val_sc)} | test={len(test_sc)}")
        log.info(f"SC Train saved → {TRAIN_SC_PATH}")
        log.info(f"SC Val   saved → {VAL_SC_PATH}")
        log.info(f"SC Test  saved → {TEST_SC_PATH}")

        log.info("Data transformation complete.")
        return train_df, val_df, test_df, feature_cols

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    train_df, val_df, test_df, feature_cols = run_transformation()

    print("\n--- TRANSFORMATION SUMMARY ---")
    print(f"Train shape : {train_df.shape}")
    print(f"Val shape   : {val_df.shape}")
    print(f"Test shape  : {test_df.shape}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  {f}")
    print(f"\nTrain target (regression) stats:")
    print(train_df[TARGET_REGRESSION].describe().round(3))
    print(f"\nTrain class balance:")
    print(train_df[TARGET_CLASSIFICATION].value_counts())
    print(f"\nNulls in train:")
    print(train_df[feature_cols].isnull().sum()[train_df[feature_cols].isnull().sum() > 0])
    if train_df[feature_cols].isnull().sum().sum() == 0:
        print("  None — all features complete.")
    

    print(f"\nSame-compound subset:")
    sc = pd.read_csv(FEATURES_SAME_COMPOUND_PATH)
    print(f"  Shape     : {sc.shape}")
    print(f"  Races     : {sc['Race'].value_counts().to_dict()}")
    print(f"  Target mean : {sc[TARGET_REGRESSION].mean():.3f}s")
    print(f"  Class balance : {sc[TARGET_CLASSIFICATION].value_counts().to_dict()}")
    print(f"\nNew feature 'tyre_life_x_coasting_delta' stats:")
    raw = pd.read_csv(FEATURES_PATH)
    print(raw["tyre_life_x_coasting_delta"].describe().round(3))