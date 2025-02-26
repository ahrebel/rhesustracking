import numpy as np
import pandas as pd
import cv2
import yaml
import argparse

def train_gaze_mapping(training_csv, output_matrix='data/trained_model/calibration_matrix_refined.npy'):
    """
    Train a refined gaze mapping using paired data from a CSV file.
    The CSV should have columns: raw_x, raw_y, touch_x, touch_y.
    """
    df = pd.read_csv(training_csv)
    raw_points = df[['raw_x', 'raw_y']].values
    touch_points = df[['touch_x', 'touch_y']].values
    H, status = cv2.findHomography(raw_points.astype(np.float32), touch_points.astype(np.float32), cv2.RANSAC)
    
    # Save the refined calibration matrix.
    np.save(output_matrix, H)
    with open('data/trained_model/calibration_refined.yaml', 'w') as f:
        yaml.dump({'homography': H.tolist()}, f)
    print("Refined calibration matrix saved to:", output_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Gaze Mapping')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    args = parser.parse_args()
    
    train_gaze_mapping(args.data)
