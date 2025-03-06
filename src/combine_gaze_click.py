#!/usr/bin/env python

"""
Combine gaze data (eye landmarks) with click data (known screen coordinates)
based on matching timestamps within a certain tolerance.

- Gaze CSV must have columns:
    time, corner_left_x, corner_left_y, corner_right_x, corner_right_y,
          left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y
  (Or whatever your script produces. Adjust column names here if needed.)

- Click file (CSV or TXT) must have columns:
    timestamp, x, y
  Each row indicates a known on-screen position (x,y) at a given timestamp.

The output CSV will have columns:
    left_corner_x, left_corner_y, right_corner_x, right_corner_y,
    left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
    screen_x, screen_y

Usage Example:
    python combine_gaze_click.py \
        --gaze_csv calibration_landmarks.csv \
        --click_file calibration_clicks.txt \
        --output_csv calibration_data_for_training.csv \
        --max_time_diff 0.1
"""

import argparse
import pandas as pd
import numpy as np
import os

def load_click_data(click_path):
    """
    Load a click/touch file which may be .csv or .txt,
    containing columns: timestamp,x,y

    We rename them to: time, screen_x, screen_y
    Returns a DataFrame with columns: time, screen_x, screen_y
    """
    # Attempt to parse the file with pandas, assuming comma-separated
    df = pd.read_csv(click_path, delimiter=',', header=0)
    
    # Rename columns if needed
    # (We assume 'timestamp' -> 'time', 'x' -> 'screen_x', 'y' -> 'screen_y')
    rename_map = {}
    if 'timestamp' in df.columns:
        rename_map['timestamp'] = 'time'
    if 'x' in df.columns:
        rename_map['x'] = 'screen_x'
    if 'y' in df.columns:
        rename_map['y'] = 'screen_y'
    
    df = df.rename(columns=rename_map)
    
    # Ensure final columns exist
    required_cols = ['time', 'screen_x', 'screen_y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in click file: {click_path}")
    
    return df

def combine_gaze_click(gaze_csv, click_file, output_csv, max_time_diff=0.05):
    """
    Merges gaze data with click data by matching each click to the nearest
    gaze entry in time, if within max_time_diff seconds.

    Gaze CSV must have columns (adjust names if needed):
        time, corner_left_x, corner_left_y, corner_right_x, corner_right_y,
              left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y

    Click file must have columns:
        time, screen_x, screen_y

    Output CSV will have:
        left_corner_x, left_corner_y, right_corner_x, right_corner_y,
        left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
        screen_x, screen_y
    """
    # Load the gaze data
    gaze_df = pd.read_csv(gaze_csv)
    
    # Basic check for required columns in the gaze data
    required_gaze_cols = [
        'time',
        'corner_left_x', 'corner_left_y',
        'corner_right_x', 'corner_right_y',
        'left_pupil_x', 'left_pupil_y',
        'right_pupil_x', 'right_pupil_y'
    ]
    for col in required_gaze_cols:
        if col not in gaze_df.columns:
            raise ValueError(f"Column '{col}' not found in gaze CSV: {gaze_csv}")

    # Load the click data (CSV or TXT)
    click_df = load_click_data(click_file)

    combined_rows = []
    for _, click_row in click_df.iterrows():
        click_time = click_row['time']
        
        # Find gaze entries within the tolerance
        candidates = gaze_df[np.abs(gaze_df['time'] - click_time) <= max_time_diff]
        if candidates.empty:
            continue
        
        # Pick the candidate with the smallest time difference
        closest_idx = (np.abs(candidates['time'] - click_time)).idxmin()
        best_match = gaze_df.loc[closest_idx]
        
        combined_rows.append({
            'left_corner_x':  best_match['corner_left_x'],
            'left_corner_y':  best_match['corner_left_y'],
            'right_corner_x': best_match['corner_right_x'],
            'right_corner_y': best_match['corner_right_y'],
            'left_pupil_x':   best_match['left_pupil_x'],
            'left_pupil_y':   best_match['left_pupil_y'],
            'right_pupil_x':  best_match['right_pupil_x'],
            'right_pupil_y':  best_match['right_pupil_y'],
            'screen_x':       click_row['screen_x'],
            'screen_y':       click_row['screen_y']
        })
    
    if not combined_rows:
        print("No matching rows found. Check your time columns or max_time_diff.")
        return
    
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined calibration data saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine gaze data with click/touch data for calibration.")
    parser.add_argument("--gaze_csv", required=True,
                        help="CSV file with raw eye landmarks (must include a 'time' column)")
    parser.add_argument("--click_file", required=True,
                        help="File (CSV or TXT) with 'timestamp,x,y' columns to be merged")
    parser.add_argument("--output_csv", required=True,
                        help="Where to save the merged calibration CSV")
    parser.add_argument("--max_time_diff", type=float, default=0.05,
                        help="Max time difference (in seconds) for matching gaze to clicks")
    args = parser.parse_args()
    
    combine_gaze_click(args.gaze_csv, args.click_file, args.output_csv, max_time_diff=args.max_time_diff)
