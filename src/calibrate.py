#!/usr/bin/env python
import cv2
import yaml
import numpy as np
import os

def load_calibration_config(config_path="config/calibration_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def compute_calibration_matrix(calib_points):
    # Extract source (eye) and destination (screen) points
    src_points = []
    dst_points = []
    for point in calib_points:
        eye = point["eye"]
        screen = point["screen"]
        src_points.append([eye["x"], eye["y"]])
        dst_points.append([screen["x"], screen["y"]])
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    # Compute perspective transform (3x3 matrix)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return matrix

def save_calibration_matrix(matrix, save_path="data/trained_model/calibration_matrix.npy"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, matrix)
    print(f"Calibration matrix saved to {save_path}")

def main():
    config = load_calibration_config()
    calib_points = config.get("calibration_points", [])
    if len(calib_points) < 4:
        print("Error: Need at least 4 calibration points.")
        return
    matrix = compute_calibration_matrix(calib_points)
    print("Calibration matrix computed:")
    print(matrix)
    save_calibration_matrix(matrix)

if __name__ == "__main__":
    main()

