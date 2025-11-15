import fastf1
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", 'data', 'cache')
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

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

        drivers = session.laps['Drivers'].nunique()
        for driver in drivers:
            try:
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
    
if __name__ == '__main__':
    session = get_session_data(2024, 'Brazil', 'R')

    if session:
        print("\nSession Info:")
        print(f"Event: {session.event['EventName']}")
        print(f"Date: {session.date}")
        print(f"Weather: {session.weather_data.head() if hasattr(session, 'weather_data') else 'N/A'}")
