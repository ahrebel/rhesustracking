#!/usr/bin/env python

"""
Combine gaze data (eye landmarks) with click data (known screen coordinates)
based on matching timestamps within a certain tolerance.
"""

import argparse
import pandas as pd
import numpy as np

def combine_gaze_click(gaze_csv, click_csv, output_csv, max_time_diff=0.05):
    """
    Merges gaze data with click data by matching each click to the nearest
    gaze entry in time, if within max_time_diff seconds.
    
    Gaze CSV must have columns:
        time, left_corner_x, left_corner_y, right_corner_x, right_corner_y,
              left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y
    
    Click CSV must have columns:
        time, screen_x, screen_y
    
    Output CSV will have:
        left_corner_x, left_corner_y, right_corner_x, right_corner_y,
        left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
        screen_x, screen_y
    """
    gaze_df = pd.read_csv(gaze_csv)
    click_df = pd.read_csv(click_csv)
    
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
            'left_corner_x': best_match['corner_left_x'],
            'left_corner_y': best_match['corner_left_y'],
            'right_corner_x': best_match['corner_right_x'],
            'right_corner_y': best_match['corner_right_y'],
            'left_pupil_x': best_match['left_pupil_x'],
            'left_pupil_y': best_match['left_pupil_y'],
            'right_pupil_x': best_match['right_pupil_x'],
            'right_pupil_y': best_match['right_pupil_y'],
            'screen_x': click_row['screen_x'],
            'screen_y': click_row['screen_y']
        })
    
    if not combined_rows:
        print("No matching rows found. Check your time columns or max_time_diff.")
        return
    
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined calibration data saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine gaze data with click data for calibration.")
    parser.add_argument("--gaze_csv", required=True, help="CSV file with raw eye landmarks and a 'time' column")
    parser.add_argument("--click_csv", required=True, help="CSV file with 'time, screen_x, screen_y' columns")
    parser.add_argument("--output_csv", required=True, help="Where to save the merged calibration CSV")
    parser.add_argument("--max_time_diff", type=float, default=0.05, 
                        help="Max time difference (in seconds) for matching gaze to clicks")
    args = parser.parse_args()
    
    combine_gaze_click(args.gaze_csv, args.click_csv, args.output_csv, max_time_diff=args.max_time_diff)
