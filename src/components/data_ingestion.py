import os
import sys
import logging
import pandas as pd
import fastf1

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    CACHE_DIR, DATA_RAW_DIR, LAPS_RAW_PATH, TELEMETRY_RAW_PATH,
    SEASON, RACES, DRIVERS, SESSION_TYPE,
    LAP_TIME_COL, LAP_NUMBER_COL, COMPOUND_COL, TYRE_LIFE_COL,
    TRACK_STATUS_COL, PIT_OUT_COL, PIT_IN_COL,
    TEL_SPEED, TEL_THROTTLE, TEL_BRAKE, TEL_GEAR, TEL_RPM, TEL_DISTANCE
)




def load_session(season, round_number, session_type):
    """Load a FastF1 session and return it."""
    try:
        log.info(f"Loading session: Season={season}, Round={round_number}, Type={session_type}")
        fastf1.Cache.enable_cache(CACHE_DIR)
        session = fastf1.get_session(season, round_number, session_type)
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        log.info(f"Session loaded: {session.event['EventName']}")
        return session
    except Exception as e:
        raise CustomException(e, sys)


def extract_clean_laps(session, drivers, race_label):
    """
    Extract clean laps for given drivers from a session.
    Returns a single DataFrame with all laps stacked.
    """
    try:
        all_laps = []

        for driver in drivers:
            laps = session.laps.pick_drivers(driver).pick_quicklaps().copy()

            # Keep only the columns we need
            cols = [
                LAP_NUMBER_COL, LAP_TIME_COL, COMPOUND_COL,
                TYRE_LIFE_COL, TRACK_STATUS_COL, PIT_OUT_COL, PIT_IN_COL
            ]
            # Only keep cols that exist in this session
            cols = [c for c in cols if c in laps.columns]
            laps = laps[cols].copy()

            laps["Driver"]     = driver
            laps["Race"]       = race_label
            laps["RoundNumber"] = session.event["RoundNumber"]

            # Convert LapTime to seconds (timedelta -> float)
            if LAP_TIME_COL in laps.columns:
                laps["LapTimeSec"] = laps[LAP_TIME_COL].dt.total_seconds()
                laps.drop(columns=[LAP_TIME_COL], inplace=True)

            # Drop pit in/out laps (NaT means no pit on that lap)
            if PIT_OUT_COL in laps.columns:
                laps = laps[laps[PIT_OUT_COL].isna()].copy()
                laps.drop(columns=[PIT_OUT_COL], inplace=True)
            if PIT_IN_COL in laps.columns:
                laps = laps[laps[PIT_IN_COL].isna()].copy()
                laps.drop(columns=[PIT_IN_COL], inplace=True)

            log.info(f"  {driver} — {len(laps)} clean laps extracted from {race_label}")
            all_laps.append(laps)

        return pd.concat(all_laps, ignore_index=True)

    except Exception as e:
        raise CustomException(e, sys)


def extract_telemetry(session, drivers, race_label):
    """
    Extract lap telemetry for given drivers.
    Returns a single DataFrame with all samples stacked.
    """
    try:
        all_tel = []

        for driver in drivers:
            laps = session.laps.pick_drivers(driver).pick_quicklaps().copy()

            for _, lap in laps.iterrows():
                try:
                    tel = lap.get_telemetry()
                except Exception:
                    # Skip laps where telemetry is unavailable
                    continue

                # Keep only the columns we need
                tel_cols = [TEL_DISTANCE, TEL_SPEED, TEL_THROTTLE, TEL_BRAKE, TEL_GEAR, TEL_RPM]
                tel_cols = [c for c in tel_cols if c in tel.columns]
                tel = tel[tel_cols].copy()

                tel["Driver"]     = driver
                tel["Race"]       = race_label
                tel["LapNumber"]  = lap[LAP_NUMBER_COL]

                all_tel.append(tel)

            log.info(f"  {driver} — telemetry extracted from {race_label}")

        return pd.concat(all_tel, ignore_index=True)

    except Exception as e:
        raise CustomException(e, sys)


def run_ingestion():
    """
    Main ingestion function.
    Loops over all races, extracts laps + telemetry, saves to data/raw/.
    """
    try:
        log.info("=" * 60)
        log.info("Starting data ingestion")
        log.info("=" * 60)

        all_laps = []
        all_tel  = []

        for round_number, race_label in RACES:
            log.info(f"\nProcessing: {race_label} (Round {round_number})")

            session = load_session(SEASON, round_number, SESSION_TYPE)

            laps = extract_clean_laps(session, DRIVERS, race_label)
            tel  = extract_telemetry(session, DRIVERS, race_label)

            all_laps.append(laps)
            all_tel.append(tel)

        # Stack all races
        laps_df = pd.concat(all_laps, ignore_index=True)
        tel_df  = pd.concat(all_tel,  ignore_index=True)

        # Save
        laps_df.to_csv(LAPS_RAW_PATH, index=False)
        log.info(f"Laps saved   → {LAPS_RAW_PATH}  | shape: {laps_df.shape}")

        tel_df.to_parquet(TELEMETRY_RAW_PATH, index=False)
        log.info(f"Telemetry saved → {TELEMETRY_RAW_PATH}  | shape: {tel_df.shape}")

        log.info("Data ingestion complete.")
        return laps_df, tel_df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    laps_df, tel_df = run_ingestion()

    print("\n--- LAPS SUMMARY ---")
    print(f"Shape     : {laps_df.shape}")
    print(f"Columns   : {list(laps_df.columns)}")
    print(f"Drivers   : {laps_df['Driver'].unique()}")
    print(f"Races     : {laps_df['Race'].unique()}")
    print(f"Nulls     :\n{laps_df.isnull().sum()}")
    print(laps_df.head(3))

    print("\n--- TELEMETRY SUMMARY ---")
    print(f"Shape     : {tel_df.shape}")
    print(f"Columns   : {list(tel_df.columns)}")
    print(f"Brake dtype: {tel_df['Brake'].dtype}")
    print(f"Nulls     :\n{tel_df.isnull().sum()}")
    print(tel_df.head(3))