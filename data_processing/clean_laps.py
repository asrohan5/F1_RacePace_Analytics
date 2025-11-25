import pandas as pd

def filter_laps(laps_df):
    required_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in required_cols:
        laps_df = laps_df[laps_df[col].notnull()]
    
    print(f"Filtered laps: {len(laps_df)} remaining")
    return laps_df

def extract_pitstop_laps(laps_df):
    pit_laps = laps_df[laps_df['PitOutTime'].notnull() | laps_df['PitInTime'].notnull()]
    print(f"Laps with pit stops: {len(pit_laps)}")
    return pit_laps

def summarize_driver_laps(laps_df):
    summary = laps_df.groupby('Driver')['LapTime'].count().reset_index()
    summary.columns = ['Driver', 'ValidLaps']
    print("\n Driver lap summary:\n", summary)
    return summary

def save_clean_data(laps_df, filename= 'clean_laps.parquet'):
    laps_df.to_parquet(filename)
    print(f"Saved cleaned laps to {filename}")

def build_race_pace(laps_df):
    df = laps_df.copy()
    df = df[df['LapTime'].notna()]

    if 'IsAccurate' in df.columns:
        df = df[df['IsAccurate'] == True]

    
    if pd.api.types.is_timedelta64_dtype(df['LapTime']):
        df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
    else:
        df['LapTimeSeconds'] = df['LapTime']

    print(f"Race pace laps: {len(df)}")
    return df


def build_pit_strategy(laps_df):
    df = laps_df.copy()
    df = df[(df['PitInTime'].notna()) | (df['PitOutTime'].notna())]

    df['HasPitIn'] = df['PitInTime'].notna()
    df['HasPitOut'] = df['PitOutTime'].notna()

    print(f'Pit-related laps: {len(df)}')
    return df

def driver_summary(race_pace_df):
    summary = (
        race_pace_df.groupby('Driver').agg(ValidLaps = ('LapTimeSeconds', 'count'), 
                                           AvgLap = ('LapTimeSeconds', 'mean'),
                                           StdLap = ('LapTimeSeconds', 'std'),
                                           BestLap = ('LapTimeSeconds', 'min')).reset_index()

    
    )
    print("Driver summary: \n", summary)
    return summary






if __name__ == '__main__':
    from data_ingestion.fetch_data import load_session_from_disk
    import os

    laps_df, metadata = load_session_from_disk(2024, 'Brazil', 'R')
    race_pace_df = build_race_pace(laps_df)
    pit_df = build_pit_strategy(laps_df)
    driver_sum = driver_summary(race_pace_df)

    os.makedirs('D:/Projects/F1ALY/data/processed', exist_ok=True)

    race_pace_df.to_parquet('D:/Projects/F1ALY/data/processed/2024_Brazil_race.parquet')
    pit_df.to_parquet('D:/Projects/F1ALY/data/processed/2024_Brazil_pit.parquet')

    driver_sum.to_parquet('D:/Projects/F1ALY/data/processed/2024_Brazil_driver_summary.parquet')

    
