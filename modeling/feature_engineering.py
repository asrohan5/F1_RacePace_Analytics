import pandas as pd
import numpy as np
from scipy import stats

def tire_features(race_pace_df, pit_df):
    
    df = race_pace_df.copy()
    df = df.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)
    
    df['TireAge'] = df.groupby('Driver').cumcount() + 1
    
    
    df['LapTimeDelta'] = df.groupby('Driver')['LapTimeSeconds'].diff()
    
    
    if 'Compound' in df.columns:
        df['Compound_Encoded'] = pd.Categorical(df['Compound'], 
                                               categories=['SOFT', 'MEDIUM', 'HARD'],
                                               ordered=True).codes
    
    print(f"Engineered tire features")
    print(f"TireAge range: {df['TireAge'].min():.0f} to {df['TireAge'].max():.0f}")
    if 'Compound' in df.columns:
        print(f"Compounds: {df['Compound'].unique().tolist()}")
    print()
    
    return df


def driver_features(race_pace_df, driver_summary_df):

    df = race_pace_df.copy()

    df= df.merge(driver_summary_df[['Driver', 'AvgLap', 'StdLap', 'BestLap']], on='Driver', how='left')
    df['PaceVsAvg'] = df['LapTimeSeconds'] - df['AvgLap']

    df['ConsistencyScore'] = 1 / (1 + df['StdLap'])

    df['GapFromBest'] = df['LapTimeSeconds'] - df['BestLap']

    print(f"Driver Features done")

    print(f" Pace v Avg range: {df['PaceVsAvg'].min(): .2f} to {df['PaceVsAvg'].max(): .2f}")
    print(f" ConsistencyScore range: {df['ConsistencyScore'].min(): .2f} to {df['ConsistencyScore'].max(): .2f}\n")

    return df

def race_context_features(race_pace_df):

    df = race_pace_df.copy()
    max_lap = df['LapNumber'].max()
    df['RacePhase'] = pd.cut( df['LapNumber'], bins=[0, max_lap * 0.33, max_lap * 0.67, max_lap], labels = ['Early', 'Mid', 'Late'])

    df['RaceProgress%'] = (df['LapNumber'] / max_lap) * 100

    print(f'Engineered race context features')
    print(f"Race phases: {df['RacePhase'].unique().tolist()}")
    print(f"MaxLaps: {max_lap}\n")

    return df

def complete_feature_set(race_pace_df, pit_df, driver_summary_df):

    print('FEATURE ENGINEER')

    df = tire_features(race_pace_df, pit_df)
    df = driver_features(df, driver_summary_df)
    df = race_context_features(df)

    feature_cols = ['Driver', 'LapNumber', 'LapTimeSeconds', 'Compound', 'Compound_Encoded', 'TireAge', 'LapTimeDelta', 'AvgLap', 'StdLap', 'BestLap', 'PaceVsAvg', 'ConsistencyScore', 'GapFromBest', 'RacePhase', 'RaceProgress%']

    df_features = df[[col for col in feature_cols if col in df.columns]].copy()
    df_features = df_features.dropna(subset=['LapTimeSeconds'])

    print(f'Built complete feature set')
    print(f'Total samples: {len(df_features)}')
    print(f'Feature: {len(df_features.columns)}\n')

    return df_features

RACES_2024 = ["Belgian Grand Prix", "Australian Grand Prix", 'Saudi Arabian Grand Prix', 'Bahrain Grand Prix']


def build_multi_race_feature_set(year=2024, races=None):
    if races is None:
        races = RACES_2024
    
    all_features = []

    for gp in races:
        print(f' Feature engineering for {year} {gp} R')
        from data_ingestion.fetch_data import load_session_from_disk
        from data_processing.clean_laps import (build_race_pace, 
                                                build_pit_strategy, 
                                                driver_summary,)

        laps_df, metadata = load_session_from_disk(year, gp, "R")

        if laps_df is None:
            continue

        race_pace_df = build_race_pace(laps_df)
        pit_df = build_pit_strategy(laps_df)
        driver_summary_df = driver_summary(race_pace_df)

        df_features = complete_feature_set(race_pace_df, pit_df, driver_summary_df)
        df_features['GrandPrix'] = gp
        df_features['Year'] = year

        all_features.append(df_features)
    
    combined = pd.concat(all_features, ignore_index = True)
    print(f'Combined feature set shape: {combined.shape}')
    return combined


if __name__ == '__main__':
    import os
    os.makedirs('D:/Projects/F1ALY/modeling', exist_ok=True)

    combined_features = build_multi_race_feature_set(year = 2024)
    combined_features.to_parquet('D:/Projects/F1ALY/modeling/2024_multi_race_features.parquet')

    print('\nSample combined data: ')
    print(combined_features.head())




