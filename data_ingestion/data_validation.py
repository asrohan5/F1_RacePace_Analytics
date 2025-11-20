import pandas as pd
import json

def validate_session_data(laps_df, metadata):
    report = {
        'status': 'PASS',
        'warnings': [],
        'errors': [],
        'summary': [],
    }

    if laps_df is None or len(laps_df) == 0:
        report['status'] = 'FAIL'
        report['errors'].append('DataFrame is empty or None')
        return report

    critical_cols = ['Driver', 'LapTime', 'Compound']
    for col in critical_cols:
        if col not in laps_df.columns:
            report['errors'].append(f'Missing Critical Column: {col}')
            report['status'] = 'FAIL'

        null_counts = laps_df.isnull().sum()
        if null_counts.sum()>0:
            report['warnings'].append(f'Found {null_counts.sum()} null values')
            print(f'Null value distribution:\n{null_counts[null_counts > 0]}')

            report['summary'] = {
                'total_rows': len(laps_df),
                'unique_drivers': laps_df['Driver'].nunique(),
                'drivers': laps_df['Driver'].unique().tolist(),
                'columns': list(laps_df.columns),
                'lap_range': f"{laps_df['LapNumber'].min()} to {laps_df['LapNumber'].max()}"

            }

            print('Data Validation Report')
            print(f"Status: {report['status']}")
            print(f"Total Rows: {report['summary']['total_rows']}")
            print(f"Unique Drivers: {report['summary']['unique_drivers']}")
            print(f"Lap Range: {report['summary']['lap_range']}")

            if report['warnings']:
                print('Warnings')
                for w in report['warnings']:
                    print(f" -{w}")
            
            if report['errors']:
                print('Errors')
                for e in report['errors']:
                    print(f" -{e}")
    return report


if __name__ == '__main__':
    from fetch_data import load_session_from_disk

    laps_df, metadata = load_session_from_disk(2024, 'Brazil', 'R')
    if laps_df is not None:
        validate_session_data(laps_df, metadata)

        
