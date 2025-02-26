#!/usr/bin/env python3
import cv2
import numpy as np
import os
import yaml
import sleap
from datetime import datetime

# Directories
VIDEO_DIR = "videos/input/"
OUTPUT_DIR = "data/trained_model/"
MODEL_DIR = "models"

# Load your trained SLEAP model.
# (Train SLEAP on your eye landmarks and save the model file, e.g. as "centered_instance_model.zip" in the models folder.)
model_path = os.path.join(MODEL_DIR, "centered_instance_model.zip")
print("Loading SLEAP model from:", model_path)
predictor = sleap.load_model(model_path)

def get_eye_coordinates(frame):
    """
    Given a frame (BGR image from OpenCV), return the eye landmark (e.g. pupil center)
    using the SLEAP predictor.
    """
    # Convert the frame to RGB (SLEAP expects RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Add batch dimension and predict
    preds = predictor.predict(img_rgb[None, ...])
    # Assume a single instance and one keypoint per instance
    instance = preds[0].instances[0]
    x = instance.points[0][0]
    y = instance.points[0][1]
    return (float(x), float(y))

def read_touch_data(file_path):
    """
    Read a touch data file (CSV with header "timestamp,x,y") and return a list of (timestamp, x, y)
    """
    data = []
    with open(file_path, "r") as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            ts_str, x_str, y_str = parts
            try:
                ts = datetime.fromisoformat(ts_str)
                x = float(x_str)
                y = float(y_str)
                data.append((ts, x, y))
            except Exception as e:
                print(f"Error parsing line: {line.strip()} - {e}")
    return data

def compute_calibration(video_file, touch_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video:", video_file)
        return None
    touch_data = read_touch_data(touch_file)
    if len(touch_data) < 4:
        print("Not enough touch data points for calibration.")
        return None
    # For calibration, pick four evenly spaced frames:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(total_frames * i / 4) for i in range(4)]
    eye_points = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        eye_pt = get_eye_coordinates(frame)
        eye_points.append(eye_pt)
    cap.release()
    if len(eye_points) < 4:
        print("Failed to get enough eye coordinate samples.")
        return None
    # Use the first 4 touch events for calibration
    touch_points = [(x, y) for (_, x, y) in touch_data[:4]]
    eye_pts_np = np.array(eye_points, dtype=np.float32)
    touch_pts_np = np.array(touch_points, dtype=np.float32)
    H, status = cv2.findHomography(eye_pts_np, touch_pts_np)
    return H

def main():
    for file in os.listdir(VIDEO_DIR):
        if file.endswith(".mp4"):
            video_path = os.path.join(VIDEO_DIR, file)
            base = os.path.splitext(file)[0]
            touch_path = os.path.join(VIDEO_DIR, base + ".txt")
            if not os.path.exists(touch_path):
                print(f"Touch file for {file} not found.")
                continue
            print(f"Calibrating {file} using {base}.txt ...")
            H = compute_calibration(video_path, touch_path)
            if H is not None:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                np.save(os.path.join(OUTPUT_DIR, f"calibration_matrix_{base}.npy"), H)
                with open(os.path.join(OUTPUT_DIR, f"calibration_{base}.yaml"), "w") as f:
                    yaml.dump(H.tolist(), f)
                print(f"Calibration successful for {file}.")
            else:
                print(f"Calibration failed for {file}.")

if __name__ == "__main__":
    main()
