#!/usr/bin/env python3
import pandas as pd
import numpy as np
import cv2
import argparse
import yaml
import os

def train_gaze_mapping(csv_file, output_dir="data/trained_model/"):
    # The CSV is expected to have columns: raw_x, raw_y, touch_x, touch_y
    df = pd.read_csv(csv_file)
    raw_pts = df[["raw_x", "raw_y"]].values.astype(np.float32)
    touch_pts = df[["touch_x", "touch_y"]].values.astype(np.float32)
    if len(raw_pts) < 4:
        print("Not enough data points for calibration.")
        return
    H, status = cv2.findHomography(raw_pts, touch_pts)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "calibration_matrix.npy"), H)
    with open(os.path.join(output_dir, "calibration.yaml"), "w") as f:
        yaml.dump(H.tolist(), f)
    print("Gaze mapping calibration updated.")

def main():
    parser = argparse.ArgumentParser(description="Train gaze mapping calibration.")
    parser.add_argument("--data", required=True, help="Path to training data CSV.")
    args = parser.parse_args()
    train_gaze_mapping(args.data)

if __name__ == "__main__":
    main()
