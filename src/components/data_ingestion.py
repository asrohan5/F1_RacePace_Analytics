import os
import sys
import pandas as pd
import fastf1

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    CACHE_DIR, DATA_RAW_DIR,
    LAPS_RAW_PATH, TELEMETRY_RAW_PATH,
    SEASON, RACES, ALL_DRIVERS, SESSION_TYPE,
    EXCLUDE_ROUNDS,
    LAP_TIME_COL, LAP_NUMBER_COL, COMPOUND_COL,
    TYRE_LIFE_COL, TRACK_STATUS_COL, PIT_OUT_COL, PIT_IN_COL,
    TEL_SPEED, TEL_THROTTLE, TEL_BRAKE, TEL_GEAR, TEL_RPM, TEL_DISTANCE,
    get_ver_upgrade_level, get_ham_upgrade_level, get_upgrade_delta
)



RACE_METADATA_PATH = os.path.join(DATA_RAW_DIR, "race_metadata.csv")


def load_session(season, round_number, session_type):
    try:
        log.info(f"  Loading session: Season={season} Round={round_number} "
                 f"Type={session_type}")
        fastf1.Cache.enable_cache(CACHE_DIR)
        session = fastf1.get_session(season, round_number, session_type)
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        log.info(f"  Session loaded: {session.event['EventName']}")
        return session
    except Exception as e:
        raise CustomException(e, sys)


def extract_clean_laps(session, drivers, race_label, round_number):
    try:
        all_laps = []

        for driver in drivers:
            try:
                laps = session.laps.pick_drivers(driver).pick_quicklaps().copy()
            except Exception:
                log.warning(f"  {driver} — no clean laps found in {race_label}, skipping")
                continue

            if len(laps) < 3:
                log.warning(f"  {driver} — only {len(laps)} laps in {race_label}, skipping")
                continue

            cols = [LAP_NUMBER_COL, LAP_TIME_COL, COMPOUND_COL,
                    TYRE_LIFE_COL, TRACK_STATUS_COL, PIT_OUT_COL, PIT_IN_COL]
            cols = [c for c in cols if c in laps.columns]
            laps = laps[cols].copy()

            laps["Driver"]      = driver
            laps["Race"]        = race_label
            laps["RoundNumber"] = round_number

            # Convert LapTime to seconds
            if LAP_TIME_COL in laps.columns:
                laps["LapTimeSec"] = laps[LAP_TIME_COL].dt.total_seconds()
                laps.drop(columns=[LAP_TIME_COL], inplace=True)

            # Drop pit laps
            if PIT_OUT_COL in laps.columns:
                laps = laps[laps[PIT_OUT_COL].isna()].copy()
                laps.drop(columns=[PIT_OUT_COL], inplace=True)
            if PIT_IN_COL in laps.columns:
                laps = laps[laps[PIT_IN_COL].isna()].copy()
                laps.drop(columns=[PIT_IN_COL], inplace=True)

            log.info(f"  {driver} — {len(laps)} clean laps from {race_label}")
            all_laps.append(laps)

        if len(all_laps) == 0:
            return None

        return pd.concat(all_laps, ignore_index=True)

    except Exception as e:
        raise CustomException(e, sys)


def extract_telemetry(session, drivers, race_label):
    try:
        all_tel = []

        for driver in drivers:
            try:
                laps = session.laps.pick_drivers(driver).pick_quicklaps().copy()
            except Exception:
                log.warning(f"  {driver} — telemetry unavailable in {race_label}")
                continue

            for _, lap in laps.iterrows():
                try:
                    tel = lap.get_telemetry()
                except Exception:
                    continue

                tel_cols = [TEL_DISTANCE, TEL_SPEED, TEL_THROTTLE,
                            TEL_BRAKE, TEL_GEAR, TEL_RPM]
                tel_cols = [c for c in tel_cols if c in tel.columns]
                tel = tel[tel_cols].copy()

                tel["Driver"]    = driver
                tel["Race"]      = race_label
                tel["LapNumber"] = lap[LAP_NUMBER_COL]

                all_tel.append(tel)

            log.info(f"  {driver} — telemetry extracted from {race_label}")

        if len(all_tel) == 0:
            return None

        return pd.concat(all_tel, ignore_index=True)

    except Exception as e:
        raise CustomException(e, sys)


def build_race_metadata():
    """
    Build a lookup table of round-level metadata.
    Includes upgrade levels and delta — used in data_transformation.py.
    """
    try:
        records = []
        for round_number, race_label in RACES:
            records.append({
                "RoundNumber"     : round_number,
                "Race"            : race_label,
                "VER_upgrade_level": get_ver_upgrade_level(round_number),
                "HAM_upgrade_level": get_ham_upgrade_level(round_number),
                "upgrade_delta"   : get_upgrade_delta(round_number),
                "is_excluded"     : int(round_number in EXCLUDE_ROUNDS),
            })
        return pd.DataFrame(records)
    except Exception as e:
        raise CustomException(e, sys)


def run_ingestion():
    try:
        log.info("=" * 60)
        log.info("Starting data ingestion — Phase 2 (all 22 rounds)")
        log.info("=" * 60)

        all_laps = []
        all_tel  = []
        skipped  = []

        for round_number, race_label in RACES:

            # Skip excluded rounds at ingestion time
            if round_number in EXCLUDE_ROUNDS:
                log.info(f"\nSkipping Round {round_number} ({race_label}) — excluded")
                skipped.append(race_label)
                continue

            log.info(f"\nProcessing: {race_label} (Round {round_number})")

            try:
                session = load_session(SEASON, round_number, SESSION_TYPE)
            except Exception as e:
                log.warning(f"  Could not load {race_label}: {e} — skipping")
                skipped.append(race_label)
                continue

            laps = extract_clean_laps(session, ALL_DRIVERS, race_label, round_number)
            tel  = extract_telemetry(session, ALL_DRIVERS, race_label)

            if laps is not None:
                all_laps.append(laps)
            if tel is not None:
                all_tel.append(tel)

        if len(all_laps) == 0:
            raise ValueError("No lap data extracted across any round.")

        laps_df = pd.concat(all_laps, ignore_index=True)
        tel_df  = pd.concat(all_tel,  ignore_index=True)

        # Save
        laps_df.to_csv(LAPS_RAW_PATH, index=False)
        log.info(f"\nLaps saved   → {LAPS_RAW_PATH} | shape: {laps_df.shape}")

        tel_df.to_parquet(TELEMETRY_RAW_PATH, index=False)
        log.info(f"Telemetry saved → {TELEMETRY_RAW_PATH} | shape: {tel_df.shape}")

        # Save race metadata lookup
        meta_df = build_race_metadata()
        meta_df.to_csv(RACE_METADATA_PATH, index=False)
        log.info(f"Metadata saved → {RACE_METADATA_PATH}")

        log.info(f"\nSkipped rounds: {skipped if skipped else 'None'}")
        log.info("Data ingestion complete.")

        return laps_df, tel_df, meta_df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    laps_df, tel_df, meta_df = run_ingestion()

    print("\n--- LAPS SUMMARY ---")
    print(f"Shape      : {laps_df.shape}")
    print(f"Drivers    : {sorted(laps_df['Driver'].unique())}")
    print(f"Races      : {sorted(laps_df['Race'].unique())}")
    print(f"Nulls      :\n{laps_df.isnull().sum()}")

    print("\n--- LAPS PER DRIVER PER RACE ---")
    print(laps_df.groupby(["Race", "Driver"]).size().unstack(fill_value=0).to_string())

    print("\n--- TELEMETRY SUMMARY ---")
    print(f"Shape      : {tel_df.shape}")
    print(f"Brake dtype: {tel_df['Brake'].dtype}")
    print(f"Nulls      :\n{tel_df.isnull().sum()}")

    print("\n--- RACE METADATA ---")
    print(meta_df.to_string(index=False))