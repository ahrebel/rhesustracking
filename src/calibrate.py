import cv2
import numpy as np
import yaml
import csv
import os

def load_touch_data(file_path):
    """
    Load touch/click data from a CSV file.
    The CSV is expected to have a header with at least the columns:
      timestamp,x,y
    """
    touch_data = []
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Parse timestamp as float and coordinates as integers
                touch_data.append({
                    "timestamp": float(row["timestamp"]),
                    "x": int(row["x"]),
                    "y": int(row["y"])
                })
            except Exception as ex:
                print("Error parsing row:", row, ex)
    return touch_data

def calibrate(video_path, touch_path):
    # Load touch data from the CSV file.
    touches = load_touch_data(touch_path)
    if len(touches) < 4:
        print("Not enough touch data points. At least 4 are required.")
        return None

    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return None

    # Read the first frame (or a designated calibration frame)
    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame from the video.")
        cap.release()
        return None

    # Define four "raw" calibration points.
    # (In a real setup these might come from the eye detection or a known target on screen.)
    h, w = frame.shape[:2]
    raw_points = np.array([[100, 100],
                           [w - 100, 100],
                           [w - 100, h - 100],
                           [100, h - 100]], dtype=np.float32)

    # Use the first four touch events as the corresponding "screen" calibration points.
    touch_points = np.array([[touches[i]["x"], touches[i]["y"]] for i in range(4)], dtype=np.float32)

    # Compute the homography that maps raw points to touch (screen) points.
    H, status = cv2.findHomography(raw_points, touch_points)
    cap.release()
    return H

if __name__ == "__main__":
    # Define folder paths (adjust if needed)
    video_folder = os.path.join("videos", "input")
    calibration_folder = os.path.join("data", "trained_model")
    os.makedirs(calibration_folder, exist_ok=True)

    # File names (ensure that 1.mp4 and 1.txt are placed in videos/input)
    video_file = os.path.join(video_folder, "1.mp4")
    touch_file = os.path.join(video_folder, "1.txt")

    # Run calibration.
    H = calibrate(video_file, touch_file)
    if H is not None:
        # Save the calibration matrix as a NumPy binary file.
        calib_npy = os.path.join(calibration_folder, "calibration_matrix.npy")
        np.save(calib_npy, H)
        # Also save to a YAML file.
        calib_yaml = os.path.join(calibration_folder, "calibration.yaml")
        with open(calib_yaml, "w") as f:
            yaml.dump({"calibration_matrix": H.tolist()}, f)
        print("Calibration successful. Matrix saved to:")
        print("  ", calib_npy)
        print("  ", calib_yaml)
    else:
        print("Calibration failed.")
