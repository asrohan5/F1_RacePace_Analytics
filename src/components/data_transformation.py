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
    STINT_LENGTH_MAP, TRACK_OVERTAKE_DIFFICULTY,
    TEST_RACE, VAL_RACES,
    THROTTLE_OFF_THRESHOLD, FULL_THROTTLE_THRESHOLD,
    MIN_BRAKE_ZONE_LENGTH_M,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
    get_upgrade_delta, RACES as RACE_LIST,
)


def load_data():
    try:
        log.info("Loading raw data...")
        laps = pd.read_csv(LAPS_RAW_PATH)
        tel = pd.read_parquet(TELEMETRY_RAW_PATH)
        meta = pd.read_csv(RACE_METADATA_PATH)
        log.info(f"Laps: {laps.shape}  Telemetry: {tel.shape}  Metadata: {meta.shape}")
        return laps, tel, meta
    except Exception as e:
        raise CustomException(e, sys)


def filter_laps(laps):
    try:
        before = len(laps)
        green = laps[laps["TrackStatus"] == 1].copy()
        log.info(f"Green flag filter: {before} to {len(green)} laps")

        green = green[~green["RoundNumber"].isin(EXCLUDE_ROUNDS)].copy()
        log.info(f"After round exclusions: {len(green)} laps")

        return green
    except Exception as e:
        raise CustomException(e, sys)


def compute_tel_features_for_lap(lap_tel):
    """
    Compute driving style metrics from raw telemetry for a single lap.
    Returns None if telemetry is too sparse to be reliable.
    """
    if len(lap_tel) < 10:
        return None

    speed = lap_tel["Speed"].values
    throttle = lap_tel["Throttle"].values
    brake = lap_tel["Brake"].astype(bool).values
    gear = lap_tel["nGear"].values
    distance = lap_tel["Distance"].values

    coasting_mask = (throttle < THROTTLE_OFF_THRESHOLD) & (~brake)
    coasting_pct = coasting_mask.mean() * 100

    full_throttle_pct = (throttle >= FULL_THROTTLE_THRESHOLD).mean() * 100

    gear_shifts = int(np.sum(np.diff(gear) != 0))

    brake_int = brake.astype(int)
    brake_diff = np.diff(brake_int, prepend=0)
    zone_starts = np.where(brake_diff == 1)[0]
    zone_ends = np.where(brake_diff == -1)[0]

    if len(zone_starts) > len(zone_ends):
        zone_ends = np.append(zone_ends, len(brake) - 1)

    brake_zone_lengths = []
    entry_speeds = []

    for s, e in zip(zone_starts, zone_ends):
        if e >= len(distance) or s >= len(distance):
            continue
        length = distance[e] - distance[s]
        if length >= MIN_BRAKE_ZONE_LENGTH_M:
            brake_zone_lengths.append(length)
            entry_speeds.append(speed[s])

    brake_zone_count = len(brake_zone_lengths)
    avg_brake_zone_length = np.mean(brake_zone_lengths) if brake_zone_lengths else 0.0
    avg_entry_speed = np.mean(entry_speeds) if entry_speeds else 0.0

    return {
        "coasting_pct": coasting_pct,
        "full_throttle_pct": full_throttle_pct,
        "gear_shifts": gear_shifts,
        "brake_zone_count": brake_zone_count,
        "avg_brake_zone_length": avg_brake_zone_length,
        "avg_entry_speed": avg_entry_speed,
    }


def compute_all_tel_features(tel):
    """
    Apply telemetry feature extraction across all (Driver, Race, LapNumber) groups.
    """
    try:
        log.info("Computing telemetry features per lap per driver...")
        records = []
        groups = tel.groupby(["Driver", "Race", "LapNumber"])
        total = len(groups)

        for i, ((driver, race, lap_number), lap_tel) in enumerate(groups):
            if i % 500 == 0:
                log.info(f"  {i}/{total} groups processed")
            feats = compute_tel_features_for_lap(lap_tel)
            if feats is None:
                continue
            feats["Driver"] = driver
            feats["Race"] = race
            feats["LapNumber"] = lap_number
            records.append(feats)

        tel_features = pd.DataFrame(records)
        log.info(f"Telemetry features: {tel_features.shape}")
        return tel_features
    except Exception as e:
        raise CustomException(e, sys)


def pair_drivers(laps_green, tel_features, meta):
    """
    Inner join VER and HAM on Race and LapNumber, then merge teammate
    race-median telemetry for style normalisation.
    """
    try:
        log.info("Pairing VER and HAM laps...")

        lap_cols = ["Driver", "Race", "RoundNumber", "LapNumber",
                    "LapTimeSec", "Compound", "TyreLife"]
        lap_cols = [c for c in lap_cols if c in laps_green.columns]
        laps_slim = laps_green[lap_cols].copy()

        combined = laps_slim.merge(tel_features, on=["Driver", "Race", "LapNumber"],
                                   how="inner")
        log.info(f"Laps merged with telemetry: {combined.shape}")

        ver_df = combined[combined["Driver"] == "VER"].copy()
        ham_df = combined[combined["Driver"] == "HAM"].copy()
        per_df = combined[combined["Driver"] == "PER"].copy()
        bot_df = combined[combined["Driver"] == "BOT"].copy()

        ver_df = ver_df[~ver_df["RoundNumber"].isin(EXCLUDE_FROM_PAIRING)].copy()
        ham_df = ham_df[~ham_df["RoundNumber"].isin(EXCLUDE_FROM_PAIRING)].copy()

        def prefix_cols(df, driver):
            rename = {
                col: f"{driver}_{col}"
                for col in df.columns
                if col not in ["Race", "RoundNumber", "LapNumber", "Driver"]
            }
            return df.drop(columns=["Driver"]).rename(columns=rename)

        ver_p = prefix_cols(ver_df, "VER")
        ham_p = prefix_cols(ham_df, "HAM")
        per_p = prefix_cols(per_df, "PER")
        bot_p = prefix_cols(bot_df, "BOT")

        paired = ver_p.merge(ham_p, on=["Race", "RoundNumber", "LapNumber"],
                             how="inner")
        log.info(f"Paired VER x HAM: {paired.shape}")

        paired = paired.merge(
            meta[["Race", "VER_upgrade_level", "HAM_upgrade_level", "upgrade_delta"]],
            on="Race", how="left"
        )

        # Teammate race medians for style normalisation.
        # Per-race median is used rather than per-lap to avoid introducing
        # lap-level noise from the teammate into the primary driver's style residual.
        per_race = per_p.groupby("Race").median(numeric_only=True).reset_index()
        bot_race = bot_p.groupby("Race").median(numeric_only=True).reset_index()

        per_race_cols = {
            c: f"PER_{c.removeprefix('PER_')}_race_med" if c.startswith("PER_") else c
            for c in per_race.columns
        }
        bot_race_cols = {
            c: f"BOT_{c.removeprefix('BOT_')}_race_med" if c.startswith("BOT_") else c
            for c in bot_race.columns
        }

        per_race = per_race.rename(columns=per_race_cols)
        bot_race = bot_race.rename(columns=bot_race_cols)

        paired = paired.merge(per_race, on="Race", how="left")
        paired = paired.merge(bot_race, on="Race", how="left")

        log.info(f"After metadata and teammate merge: {paired.shape}")
        return paired

    except Exception as e:
        raise CustomException(e, sys)


def engineer_features(paired):
    """
    Build all delta, interaction, and contextual features from the paired lap data.
    """
    try:
        log.info("Engineering features...")
        df = paired.copy()

        # Raw deltas: VER minus HAM for each telemetry metric.
        # Negative delta means HAM has more of that behaviour on that lap.
        df["coasting_pct_delta"] = df["VER_coasting_pct"] - df["HAM_coasting_pct"]
        df["full_throttle_pct_delta"] = df["VER_full_throttle_pct"] - df["HAM_full_throttle_pct"]
        df["gear_shifts_delta"] = df["VER_gear_shifts"] - df["HAM_gear_shifts"]
        df["avg_brake_zone_length_delta"] = df["VER_avg_brake_zone_length"] - df["HAM_avg_brake_zone_length"]
        df["avg_entry_speed_delta"] = df["VER_avg_entry_speed"] - df["HAM_avg_entry_speed"]
        df["brake_zone_count_delta"] = df["VER_brake_zone_count"] - df["HAM_brake_zone_count"]

        df["tyre_life_delta"] = df["VER_TyreLife"] - df["HAM_TyreLife"]

        # Tyre age times coasting captures whether the driver with older tyres
        # is coasting more (managing) or less (pushing). Clipped to suppress extremes.
        df["tyre_life_x_coasting_delta"] = (
            df["tyre_life_delta"] * df["coasting_pct_delta"]
        ).clip(-50, 50)

        # Stint phase normalises tyre life by the expected compound stint length,
        # giving a 0-1 measure of how far into the stint each driver is.
        def stint_phase(tyre_life, compound):
            expected = compound.map(STINT_LENGTH_MAP).fillna(25)
            return tyre_life / expected

        df["VER_stint_phase"] = stint_phase(df["VER_TyreLife"], df["VER_Compound"])
        df["HAM_stint_phase"] = stint_phase(df["HAM_TyreLife"], df["HAM_Compound"])
        df["stint_phase_delta"] = df["VER_stint_phase"] - df["HAM_stint_phase"]

        # Compound encoding
        compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
                        "INTERMEDIATE": 3, "WET": 4}
        df["VER_compound_enc"] = df["VER_Compound"].map(compound_map).fillna(-1).astype(int)
        df["HAM_compound_enc"] = df["HAM_Compound"].map(compound_map).fillna(-1).astype(int)
        df["same_compound"] = (df["VER_Compound"] == df["HAM_Compound"]).astype(int)

        df["is_low_sample"] = df["RoundNumber"].isin(LOW_SAMPLE_ROUNDS).astype(int)

        # Teammate-normalised style features.
        # Driver metric minus teammate race median, both in the same car.
        # This isolates driving behaviour from car pace.
        # Capped at +-5 to handle races where the teammate had anomalous laps
        # (e.g. Bottas wing failure at Imola distorts the median).
        def safe_style(driver_col, teammate_col, df):
            return (df[driver_col] - df[teammate_col]).clip(-5, 5)

        df["VER_style_coasting"] = safe_style(
            "VER_coasting_pct", "PER_coasting_pct_race_med", df)
        df["HAM_style_coasting"] = safe_style(
            "HAM_coasting_pct", "BOT_coasting_pct_race_med", df)

        excluded_tmm_races = df["Race"].isin(
            df[df["RoundNumber"].isin(EXCLUDE_FROM_TEAMMATE)]["Race"].unique()
        )
        df.loc[excluded_tmm_races, "VER_style_coasting"] = 0
        df.loc[excluded_tmm_races, "HAM_style_coasting"] = 0
        df["style_coasting_delta"] = df["VER_style_coasting"] - df["HAM_style_coasting"]

        df["VER_style_brake_length"] = safe_style(
            "VER_avg_brake_zone_length", "PER_avg_brake_zone_length_race_med", df)
        df["HAM_style_brake_length"] = safe_style(
            "HAM_avg_brake_zone_length", "BOT_avg_brake_zone_length_race_med", df)
        df.loc[excluded_tmm_races, "VER_style_brake_length"] = 0
        df.loc[excluded_tmm_races, "HAM_style_brake_length"] = 0
        df["style_brake_delta"] = df["VER_style_brake_length"] - df["HAM_style_brake_length"]

        df["VER_style_full_throttle"] = safe_style(
            "VER_full_throttle_pct", "PER_full_throttle_pct_race_med", df)
        df["HAM_style_full_throttle"] = safe_style(
            "HAM_full_throttle_pct", "BOT_full_throttle_pct_race_med", df)
        df.loc[excluded_tmm_races, "VER_style_full_throttle"] = 0
        df.loc[excluded_tmm_races, "HAM_style_full_throttle"] = 0
        df["style_throttle_delta"] = df["VER_style_full_throttle"] - df["HAM_style_full_throttle"]

        # Targets
        df[TARGET_REGRESSION] = df["VER_LapTimeSec"] - df["HAM_LapTimeSec"]
        df[TARGET_CLASSIFICATION] = (df[TARGET_REGRESSION] < 0).astype(int)

        # Normalised lap number within each race (0 to 1).
        # Captures race phase without encoding the target or race identity.
        df = df.sort_values(["Race", "LapNumber"]).reset_index(drop=True)
        df["lap_race_position"] = (
            df.groupby("Race")["LapNumber"]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
        )

        # Upgrade delta EMA: exponentially weighted moving average of the
        # step-function upgrade delta across the season. The EMA captures
        # that car performance advantages accumulate and decay smoothly
        # rather than switching on at a specific round number.
        round_order = [r for r, _ in sorted(RACE_LIST, key=lambda x: x[0])]
        ema_map = {}
        ema_val = None
        alpha = 2 / (3 + 1)
        for rnd in round_order:
            raw = get_upgrade_delta(rnd)
            if ema_val is None:
                ema_val = float(raw)
            else:
                ema_val = alpha * raw + (1 - alpha) * ema_val
            ema_map[rnd] = round(ema_val, 4)
        df["upgrade_delta_ema"] = df["RoundNumber"].map(ema_map).fillna(0.0)

        # Stint phase times coasting delta resolves the directional ambiguity
        # in coasting_pct_delta. A driver coasting more when deep into a stint
        # is managing tyres (positive signal); coasting more on fresh tyres
        # suggests caution rather than pace advantage.
        df["stint_phase_x_coasting_delta"] = (
            df["stint_phase_delta"] * df["coasting_pct_delta"]
        ).clip(-5, 5)

        # Track overtake difficulty from config. Circuits with high scores have
        # more frequent position changes and stronger DRS effects, which means
        # the lap time delta is more confounded by track position than by style.
        df["track_overtake_difficulty"] = df["Race"].map(
            TRACK_OVERTAKE_DIFFICULTY
        ).fillna(2).astype(int)

        # sc_rolling_delta_3: 3-lap rolling mean of the target on same-compound laps.
        # shift(1) prevents current-lap leakage. Not in FEATURE_COLS — retained
        # here for potential Phase 3 use and EDA reference only.
        sc_mask = df["VER_Compound"] == df["HAM_Compound"]
        df["sc_rolling_delta_3"] = np.nan
        df.loc[sc_mask, "sc_rolling_delta_3"] = (
            df[sc_mask]
            .groupby("Race")[TARGET_REGRESSION]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        df["sc_rolling_delta_3"] = df["sc_rolling_delta_3"].fillna(0.0)

        log.info(f"Features engineered. Shape: {df.shape}")
        return df

    except Exception as e:
        raise CustomException(e, sys)


FEATURE_COLS = [
    # Raw deltas — relative VER minus HAM for each telemetry metric.
    # Individual per-driver raw values excluded: they are circuit-level
    # confounders (both drivers go faster on fast circuits) and the
    # correlation between VER and HAM raw values is 0.7-0.9 for most
    # metrics. The delta captures the relevant relative signal cleanly.
    "coasting_pct_delta",
    "full_throttle_pct_delta",
    "gear_shifts_delta",
    "avg_brake_zone_length_delta",
    "avg_entry_speed_delta",
    "brake_zone_count_delta",
    "tyre_life_delta",
    "tyre_life_x_coasting_delta",
    "stint_phase_delta",

    # Race phase: normalised lap number within the race (0 to 1).
    "lap_race_position",

    # Car development context.
    # upgrade_delta_ema replaces the raw step-function upgrade_delta as the
    # primary car context signal. The EMA reduces era-level prediction bias
    # by smoothing the discrete jump at each upgrade round.
    # upgrade_delta retained alongside it as the step-function reference.
    "upgrade_delta",
    "upgrade_delta_ema",
    "is_low_sample",

    # Teammate-normalised driving style residuals.
    # Each value is the driver's metric minus their teammate's race median
    # on the same equipment, isolating driving behaviour from car pace.
    "VER_style_coasting",
    "HAM_style_coasting",
    "style_coasting_delta",
    "VER_style_brake_length",
    "HAM_style_brake_length",
    "style_brake_delta",
    "VER_style_full_throttle",
    "HAM_style_full_throttle",
    "style_throttle_delta",

    # Interaction of stint phase and coasting delta.
    # Contextualises the direction of coasting behaviour by where each
    # driver is in their tyre life.
    "stint_phase_x_coasting_delta",

    # Circuit character: how strongly race position and DRS effects
    # dominate the lap time delta relative to pure driving style.
    # Replaces race_enc (round number) which allowed the model to
    # memorise race-specific outcomes rather than learn transferable
    # style patterns.
    "track_overtake_difficulty",

    # Compound context
    "same_compound",
    "VER_compound_enc",
    "HAM_compound_enc",
]

ID_COLS = ["Race", "RoundNumber", "LapNumber",
           TARGET_REGRESSION, TARGET_CLASSIFICATION]


def select_columns(df):
    try:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            log.warning(f"Missing feature columns: {missing}")

        present_features = [c for c in FEATURE_COLS if c in df.columns]
        final_cols = ID_COLS + present_features

        result = df[final_cols].copy()
        before = len(result)
        result = result.dropna(subset=present_features + [TARGET_REGRESSION])
        log.info(f"Dropped {before - len(result)} rows with NaN in features or target")
        log.info(f"Final feature dataframe: {result.shape}")
        return result, present_features

    except Exception as e:
        raise CustomException(e, sys)


def loro_split(df):
    """
    Leave-one-race-out split. No lap from one race ever appears in another split.
    Test is AbuDhabi (held out entirely). Val is Brazil and Qatar.
    """
    try:
        log.info("Performing Leave-One-Race-Out split...")

        test_df = df[df["Race"] == TEST_RACE].copy()
        val_df = df[df["Race"].isin(VAL_RACES)].copy()
        train_df = df[~df["Race"].isin([TEST_RACE] + VAL_RACES)].copy()

        log.info(f"Train: {len(train_df)} laps from {sorted(train_df['Race'].unique())}")
        log.info(f"Val: {len(val_df)} laps from {sorted(val_df['Race'].unique())}")
        log.info(f"Test: {len(test_df)} laps from {sorted(test_df['Race'].unique())}")

        return train_df, val_df, test_df

    except Exception as e:
        raise CustomException(e, sys)


def build_sc_subset(df):
    """
    Same-compound subset: only laps where both drivers are on identical tyre compounds.
    Removes compound strategy confounding from the style signal.
    """
    try:
        log.info("Building same-compound subset...")

        sc = df[df["same_compound"] == 1].copy()
        sc = sc[~sc["RoundNumber"].isin(EXCLUDE_FROM_SC)].copy()

        log.info(f"SC subset: {len(sc)} rows ({100*len(sc)/len(df):.1f}% of full dataset)")

        sc_test = sc[sc["Race"] == TEST_RACE].copy()
        sc_val = sc[sc["Race"].isin(VAL_RACES)].copy()
        sc_train = sc[~sc["Race"].isin([TEST_RACE] + VAL_RACES)].copy()

        log.info(f"SC train: {len(sc_train)}  val: {len(sc_val)}  test: {len(sc_test)}")

        return sc, sc_train, sc_val, sc_test

    except Exception as e:
        raise CustomException(e, sys)


def scale_splits(train_df, val_df, test_df,
                 sc_train_df, sc_val_df, sc_test_df,
                 feature_cols):
    """
    Fit StandardScaler on training data only, then apply to all splits.
    Prevents any information from val or test influencing the scaling parameters.
    """
    try:
        log.info("Scaling features (fit on train only)...")

        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols])

        def apply_scale(df):
            scaled = df.copy()
            scaled[feature_cols] = scaler.transform(df[feature_cols])
            return scaled

        train_scaled = apply_scale(train_df)
        val_scaled = apply_scale(val_df)
        test_scaled = apply_scale(test_df)
        sc_train_scaled = apply_scale(sc_train_df)
        sc_val_scaled = apply_scale(sc_val_df)
        sc_test_scaled = apply_scale(sc_test_df)

        log.info("Scaler fitted and applied.")
        return (scaler,
                train_scaled, val_scaled, test_scaled,
                sc_train_scaled, sc_val_scaled, sc_test_scaled)

    except Exception as e:
        raise CustomException(e, sys)


def save_outputs(features_df, sc_features_df,
                 train_scaled, val_scaled, test_scaled,
                 sc_train_scaled, sc_val_scaled, sc_test_scaled,
                 scaler):
    try:
        log.info("Saving outputs...")

        features_df.to_csv(FEATURES_PATH, index=False)
        log.info(f"features.csv: {features_df.shape}")

        sc_features_df.to_csv(FEATURES_SAME_COMPOUND_PATH, index=False)
        log.info(f"features_same_compound.csv: {sc_features_df.shape}")

        train_scaled.to_csv(TRAIN_PATH, index=False)
        val_scaled.to_csv(VAL_PATH, index=False)
        test_scaled.to_csv(TEST_PATH, index=False)
        log.info(f"train: {train_scaled.shape}  val: {val_scaled.shape}  test: {test_scaled.shape}")

        sc_train_scaled.to_csv(TRAIN_SC_PATH, index=False)
        sc_val_scaled.to_csv(VAL_SC_PATH, index=False)
        sc_test_scaled.to_csv(TEST_SC_PATH, index=False)
        log.info(f"SC train: {sc_train_scaled.shape}  val: {sc_val_scaled.shape}  test: {sc_test_scaled.shape}")

        with open(PREPROCESSOR_PATH, "wb") as f:
            pickle.dump(scaler, f)
        log.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    except Exception as e:
        raise CustomException(e, sys)


def run_transformation():
    try:
        log.info("Starting data transformation")

        laps, tel, meta = load_data()
        laps_green = filter_laps(laps)
        tel_features = compute_all_tel_features(tel)
        paired = pair_drivers(laps_green, tel_features, meta)
        features_full = engineer_features(paired)
        features_df, feature_cols = select_columns(features_full)
        train_df, val_df, test_df = loro_split(features_df)
        sc_df, sc_train_df, sc_val_df, sc_test_df = build_sc_subset(features_df)

        (scaler,
         train_scaled, val_scaled, test_scaled,
         sc_train_scaled, sc_val_scaled, sc_test_scaled) = scale_splits(
            train_df, val_df, test_df,
            sc_train_df, sc_val_df, sc_test_df,
            feature_cols
        )

        save_outputs(
            features_df, sc_df,
            train_scaled, val_scaled, test_scaled,
            sc_train_scaled, sc_val_scaled, sc_test_scaled,
            scaler
        )

        log.info("Data transformation complete.")

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

    log.info("Full dataset")
    log.info(f"  Total rows: {len(features_df)}")
    log.info(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    log.info(f"  Target mean: {features_df[TARGET_REGRESSION].mean():.3f}s")
    log.info(f"  Target std: {features_df[TARGET_REGRESSION].std():.3f}s")
    log.info(f"  Class balance: {features_df[TARGET_CLASSIFICATION].mean():.2%} VER faster laps")

    log.info("SC dataset")
    log.info(f"  Total SC rows: {len(sc_df)}")
    log.info(f"  SC train: {len(sc_train)}  val: {len(sc_val)}  test: {len(sc_test)}")

    null_counts = features_df.isnull().sum()
    if null_counts.any():
        log.info("Feature nulls:")
        log.info(null_counts[null_counts > 0].to_string())
    else:
        log.info("No nulls in feature dataframe.")

    log.info("Split sanity check:")
    train_races = set(train["Race"].unique())
    val_races = set(val["Race"].unique())
    test_races = set(test["Race"].unique())
    log.info(f"  train/val overlap: {train_races & val_races}")
    log.info(f"  train/test overlap: {train_races & test_races}")
    log.info(f"  val/test overlap: {val_races & test_races}")