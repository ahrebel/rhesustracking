import cv2
import numpy as np
import yaml
import csv
import os
from dateutil import parser  # For robust ISO8601 timestamp parsing

def load_touch_data(file_path):
    """
    Load touch/click data from a CSV file.
    The CSV is expected to have a header with columns:
      timestamp,x,y
    """
    touch_data = []
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Parse the timestamp using dateutil (returns a datetime; then convert to float seconds)
                ts = parser.isoparse(row["timestamp"]).timestamp()
                touch_data.append({
                    "timestamp": ts,
                    "x": int(row["x"]),
                    "y": int(row["y"])
                })
            except Exception as ex:
                print("Error parsing row:", row, ex)
    return touch_data

def calibrate(video_path, touch_path):
    touches = load_touch_data(touch_path)
    if len(touches) < 4:
        print(f"Not enough touch data points in {touch_path}. At least 4 are required.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return None

    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame from the video:", video_path)
        cap.release()
        return None

    h, w = frame.shape[:2]
    # Define four "raw" calibration points (this is an example; you might use detected eye positions in a real system)
    raw_points = np.array([[100, 100],
                           [w - 100, 100],
                           [w - 100, h - 100],
                           [100, h - 100]], dtype=np.float32)
    # Use the first four touch events as the corresponding "screen" calibration points.
    touch_points = np.array([[touches[i]["x"], touches[i]["y"]] for i in range(4)], dtype=np.float32)

    H, status = cv2.findHomography(raw_points, touch_points)
    cap.release()
    return H

def process_all_calibrations(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = os.listdir(input_folder)
    # Look for video files named with a number and .mp4 extension.
    video_files = [f for f in files if f.lower().endswith(".mp4") and f[:-4].isdigit()]
    video_files.sort(key=lambda x: int(x[:-4]))
    for video_file in video_files:
        base = os.path.splitext(video_file)[0]
        touch_file = base + ".txt"
        video_path = os.path.join(input_folder, video_file)
        touch_path = os.path.join(input_folder, touch_file)
        calib_npy = os.path.join(output_folder, f"calibration_matrix_{base}.npy")
        calib_yaml = os.path.join(output_folder, f"calibration_{base}.yaml")
        if not os.path.exists(touch_path):
            print(f"Touch file {touch_path} not found, skipping {video_file}.")
            continue
        if os.path.exists(calib_npy):
            print(f"Calibration for {base} already exists, skipping.")
            continue
        print(f"Calibrating {video_file} using {touch_file}...")
        H = calibrate(video_path, touch_path)
        if H is not None:
            np.save(calib_npy, H)
            with open(calib_yaml, "w") as f:
                yaml.dump({"calibration_matrix": H.tolist()}, f)
            print(f"Calibration saved for {base}.")
        else:
            print(f"Calibration failed for {base}.")

if __name__ == "__main__":
    input_folder = os.path.join("videos", "input")
    output_folder = os.path.join("data", "trained_model")
    process_all_calibrations(input_folder, output_folder)
