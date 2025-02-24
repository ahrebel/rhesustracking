#!/usr/bin/env python
import numpy as np
import cv2
import pandas as pd
import argparse
import os

def train_gaze_mapping(training_csv, output_matrix="data/trained_model/calibration_matrix.npy"):
    """
    Trains a new gaze mapping (calibration) matrix using paired data.
    The CSV file should contain the following columns:
      - raw_x, raw_y: Raw eye coordinates (from video frames)
      - touch_x, touch_y: Corresponding touch (ground truth) coordinates
    Uses RANSAC to compute a robust homography matrix.
    """
    df = pd.read_csv(training_csv)
    required_columns = ['raw_x', 'raw_y', 'touch_x', 'touch_y']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain columns: raw_x, raw_y, touch_x, touch_y")
    
    # Prepare source (raw eye coordinates) and destination (touch coordinates) points
    src_points = df[['raw_x', 'raw_y']].values.astype(np.float32)
    dst_points = df[['touch_x', 'touch_y']].values.astype(np.float32)
    
    # Compute homography using RANSAC
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    if H is None:
        raise RuntimeError("Could not compute homography. Check your training data.")
    
    os.makedirs(os.path.dirname(output_matrix), exist_ok=True)
    np.save(output_matrix, H)
    print(f"Calibration matrix saved to {output_matrix}")
    print("Calibration matrix:")
    print(H)

def main():
    parser = argparse.ArgumentParser(description="Train gaze mapping from paired raw eye and touch data.")
    parser.add_argument("--data", required=True, help="Path to the CSV file containing training data.")
    args = parser.parse_args()
    train_gaze_mapping(args.data)

if __name__ == "__main__":
    main()
