
#!/usr/bin/env python

"""
Combine gaze data (eye landmarks) with click/touch data (known screen coordinates)
based on matching timestamps within a certain tolerance.

- Gaze CSV must have columns (adjust if needed):
    time, corner_left_x, corner_left_y, corner_right_x, corner_right_y,
          left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y

- Click file can be .csv or .txt, with columns:
    timestamp,x,y
  (We'll rename 'timestamp' -> 'time', 'x' -> 'screen_x', 'y' -> 'screen_y'.)

We parse any non-numeric time columns as datetimes and convert them to "seconds from
the earliest time in that file". The script then matches each click row to the nearest
gaze row within `max_time_diff` seconds.

Output CSV will have columns:
    left_corner_x, left_corner_y, right_corner_x, right_corner_y,
    left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
    screen_x, screen_y

If the output CSV already exists, new matches are **appended** to the existing file.

Usage Example:
    python combine_gaze_click.py \
        --gaze_csv landmarks_output.csv \
        --click_file /Users/anthonyrebello/rhesustracking/videos/input/3.txt \
        --output_csv calibration_data_for_training.csv \
        --max_time_diff 0.1
"""

import argparse
import pandas as pd
import numpy as np
import os

def parse_time_column(df, time_col='time'):
    """
    Attempt to parse the given 'time_col' in df.
    - If it's already numeric, keep it as-is (float).
    - Otherwise, parse as datetime and convert to seconds from the earliest time in df.
    Returns the DataFrame with a new column 'time_sec' that is float.
    """
    if df[time_col].dtype.kind in ('i', 'f'):
        # Already numeric
        df['time_sec'] = df[time_col].astype(float)
    else:
        # Parse as datetime
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if df[time_col].isnull().any():
            raise ValueError(f"Some values in column '{time_col}' could not be parsed as datetimes.")
        # Convert to seconds from earliest timestamp in this file
        earliest = df[time_col].min()
        df['time_sec'] = (df[time_col] - earliest).dt.total_seconds()
    return df

def load_click_data(click_path):
    """
    Load a click/touch file (CSV or TXT), containing columns: timestamp,x,y
    We'll rename them to: time, screen_x, screen_y.
    Then parse the time column into numeric seconds in 'time_sec'.
    """
    # Read the file, assuming comma-separated
    df = pd.read_csv(click_path, delimiter=',', header=0)
    
    # Rename columns if needed
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
    
    # Parse the 'time' column -> 'time_sec'
    df = parse_time_column(df, time_col='time')
    return df

def combine_gaze_click(gaze_csv, click_file, output_csv, max_time_diff=0.05):
    """
    Merges gaze data with click data by matching each click to the nearest
    gaze entry in time, if within max_time_diff seconds.

    Gaze CSV must have columns (adjust names if needed):
        time, corner_left_x, corner_left_y, corner_right_x, corner_right_y,
              left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y
    We'll parse the 'time' column into 'time_sec'.

    Click file must have columns: time, screen_x, screen_y (after rename)
    We'll parse the 'time' column into 'time_sec' as well.

    Output CSV will have:
        left_corner_x, left_corner_y, right_corner_x, right_corner_y,
        left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
        screen_x, screen_y

    If output_csv already exists, new matches are appended to it.
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

    # Parse gaze time column -> 'time_sec'
    gaze_df = parse_time_column(gaze_df, time_col='time')

    # Load the click data (CSV or TXT) -> parse time -> 'time_sec'
    click_df = load_click_data(click_file)

    combined_rows = []
    for _, click_row in click_df.iterrows():
        click_time_sec = click_row['time_sec']
        
        # Find gaze entries within the tolerance
        # i.e., where abs(gaze_time_sec - click_time_sec) <= max_time_diff
        candidates = gaze_df[np.abs(gaze_df['time_sec'] - click_time_sec) <= max_time_diff]
        if candidates.empty:
            continue
        
        # Pick the candidate with the smallest time difference
        closest_idx = (np.abs(candidates['time_sec'] - click_time_sec)).idxmin()
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
    
    # Create DataFrame of new matches
    new_matches_df = pd.DataFrame(combined_rows)
    
    # If output file already exists, append to it
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        # Concatenate old + new
        combined_df = pd.concat([existing_df, new_matches_df], ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"Appended {len(new_matches_df)} new rows to existing file '{output_csv}'.")
    else:
        new_matches_df.to_csv(output_csv, index=False)
        print(f"Combined calibration data saved to new file '{output_csv}'.")

    # Scale the screen_x and screen_y columns by 2
    new_matches_df['screen_x'] /= 2
    new_matches_df['screen_y'] /= 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine gaze data with click/touch data for calibration.")
    parser.add_argument("--gaze_csv", required=True,
                        help="CSV file with raw eye landmarks (must include a 'time' column)")
    parser.add_argument("--click_file", required=True,
                        help="File (CSV or TXT) with 'timestamp,x,y' columns to be merged")
    parser.add_argument("--output_csv", required=True,
                        help="Where to save the merged calibration CSV (appends if file exists)")
    parser.add_argument("--max_time_diff", type=float, default=0.05,
                        help="Max time difference (in seconds) for matching gaze to clicks")
    args = parser.parse_args()
    
    combine_gaze_click(args.gaze_csv, args.click_file, args.output_csv, max_time_diff=args.max_time_diff)
