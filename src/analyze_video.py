#!/usr/bin/env python3
import cv2
import numpy as np
import os
import yaml
import sleap

# Directories
VIDEO_DIR = "videos/input/"
OUTPUT_DIR = "data/analysis_output/"
CALIB_DIR = "data/trained_model/"
MODEL_DIR = "models"

# Load your SLEAP model
model_path = os.path.join(MODEL_DIR, "centered_instance_model.zip")
print("Loading SLEAP model from:", model_path)
predictor = sleap.load_model(model_path)

def get_eye_coordinates(frame):
    """
    Uses the SLEAP predictor to return the eye landmark (pupil center) from the frame.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preds = predictor.predict(img_rgb[None, ...])
    # Assume a single instance and a single keypoint (the pupil)
    instance = preds[0].instances[0]
    x = instance.points[0][0]
    y = instance.points[0][1]
    return (float(x), float(y))

def load_calibration_matrix(video_file):
    base = os.path.splitext(os.path.basename(video_file))[0]
    calib_path = os.path.join(CALIB_DIR, f"calibration_matrix_{base}.npy")
    if os.path.exists(calib_path):
        H = np.load(calib_path)
        return H
    else:
        print("No calibration matrix found for", video_file)
        return None

def process_video(video_file):
    H = load_calibration_matrix(video_file)
    if H is None:
        print("Skipping video due to missing calibration.")
        return
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video:", video_file)
        return

    gaze_data = []  # Will hold tuples of (timestamp, calibrated_x, calibrated_y)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Time in seconds (approximate)
        ts = frame_idx / fps
        raw_coords = get_eye_coordinates(frame)
        # Use homography to map raw eye coordinates to calibrated screen coordinates
        raw_pt = np.array([[[raw_coords[0], raw_coords[1]]]], dtype=np.float32)
        mapped_pt = cv2.perspectiveTransform(raw_pt, H)
        calibrated_x, calibrated_y = mapped_pt[0][0]
        gaze_data.append((ts, calibrated_x, calibrated_y))
        frame_idx += 1

    cap.release()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_file))[0]
    output_csv = os.path.join(OUTPUT_DIR, f"gaze_{base}.csv")
    with open(output_csv, "w") as f:
        f.write("timestamp,calibrated_x,calibrated_y\n")
        for ts, x, y in gaze_data:
            f.write(f"{ts},{x},{y}\n")
    print(f"Analysis complete for {video_file}. Gaze data saved to {output_csv}")

def main():
    for file in os.listdir(VIDEO_DIR):
        if file.endswith(".mp4"):
            video_path = os.path.join(VIDEO_DIR, file)
            print("Analyzing video:", video_path)
            process_video(video_path)

if __name__ == "__main__":
    main()
