import fastf1
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import os

CACHE_DIR = os.path.join("D:/Projects/F1ALY", 'data', 'cache')
RAW_DATA_DIR = os.path.join("D:/Projects/F1ALY", 'data', 'raw')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok = True)

fastf1.Cache.enable_cache(CACHE_DIR)

def get_session_data(year, grand_prix, session_type):
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load()

        print(f"Successfully leader {year} {grand_prix} {session_type}")
        print(f"    - Total Laps: {len(session.laps)}")
        print(f"    - Drivers: {session.laps['Driver'].nunique()}")

        return session
    except Exception as e:
        print(f" Error loading session {e}")
        return None
    
def save_session(session, year, grand_prix, session_type):

    if session.laps.empty:
        print("No laps data for this session.")
        return None


    if session is None:
        print('Session is None, cannot save')
        return
    
    try:
        event_dir = os.path.join(RAW_DATA_DIR, f"{year}_{grand_prix}_{session_type}")
        os.makedirs(event_dir, exist_ok=True)

        laps_file = os.path.join(event_dir, 'laps.parquet')
        session.laps.to_parquet(laps_file)

        telemetry_dir = os.path.join(event_dir, 'telemetry')
        os.makedirs(telemetry_dir, exist_ok=True)

        #drivers = session.laps['Drivers'].nunique()
        drivers = session.laps['Driver'].dropna().unique()

        for driver in drivers:
            try:
                
                if session.laps[session.laps['Driver'] == driver].empty:
                    print(f"No lap data for driver {driver}, skipping telemetry save.")
                    continue

                driver_telemetry = session.laps[session.laps['Driver']==driver].get_telemetry().reset_index(drop=True)
                if len(driver_telemetry) > 0:
                    telemetry_file = os.path.join(telemetry_dir, f"{driver}_telemetry.parquet")
                    driver_telemetry.to_parquet(telemetry_file)

            except Exception as e:
                print(f'Couldnot save telemetry for {driver}: {e}')
        
        metadata = {
            'year': year,
            'grand_prix': grand_prix,
            'session_type': session_type,
            'date': str(session.date), 
            'num_drivers': len(drivers), 
            'num_laps': len(session.laps),
            'saved_at': datetime.now().isoformat()

        }

        metadata_file = os.path.join(event_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent = 4)
        print(f"data saved to {event_dir}")

        return event_dir

    except Exception as e:
        print(f"error saving session data: {e}")
        return None


def batch_fetch_sessions(session_list):
    results = { 
        'successful': [],
        'failed': []
    }
    for year, grand_prix, session_type in session_list:
        print(f"\n Fetching {year} {grand_prix} {session_type}")

        session = get_session_data(year, grand_prix, session_type)

        if session is not None:
            save_path = save_session(session, year, grand_prix, session_type)
            if save_path:
                results['successful'].append((year, grand_prix, session_type))
        else:
            results['failed'].append((year, grand_prix, session_type))

        print("Batch Fetch Summary")
        print(f"Successful: {len(results['successful'])}")
        for item in results['successful']:
            print(f" - {item}")
        print(f" Failed: {len(results['failed'])}")
        for item in results['failed']:
            print(f" - {item}")
        
        return results

def load_session_from_disk(year, grand_prix, session_type):
    event_dir = os.path.join(RAW_DATA_DIR, f"{year}_{grand_prix}_{session_type}")

    if not os.path.exists(event_dir):
        print(f" Data not found for {year} {grand_prix} {session_type}")
        return None, None
    
    try:
        laps_file = os.path.join(event_dir, 'laps.parquet')
        laps_df = pd.read_parquet(laps_file)

        metadata_file = os.path.join(event_dir, 'metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded {year} {grand_prix} {session_type} from disk")
        print(f" - Rows: {len(laps_df)}")
        print(f" - Columns: {len(laps_df.columns)}")

        return laps_df, metadata
    except Exception as e:
        print(f'Error loading data: {e}')
        return None, None




    
if __name__ == '__main__':

    sessions_to_fetch = [
        (2024, 'Brazil', 'R'),
        (2024, 'Belgium', 'R'),
        (2024, 'Australia', 'R')
    ]

    results = batch_fetch_sessions(sessions_to_fetch)

    print('Testing disk Load')
    laps_df, metadata = load_session_from_disk(2024, 'Brazil', 'R')

    if laps_df is not None:
        print(f"\n Laps Dataframe Info:")
        print(f"Shape: {laps_df.shape}")
        print(f"\n First few rows:")
        print(laps_df.head())
