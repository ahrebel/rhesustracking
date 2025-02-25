import cv2
import numpy as np
import yaml
import csv
import os

def load_touch_data(file_path):
    """
    Load touch/click data from a CSV or TXT file.
    The file is expected to have a header with at least the columns:
      timestamp,x,y
    (Here we keep the timestamp as a string since ISO timestamps are not directly numeric.)
    """
    touch_data = []
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                touch_data.append({
                    "timestamp": row["timestamp"],
                    "x": int(row["x"]),
                    "y": int(row["y"])
                })
            except Exception as ex:
                print("Error parsing row:", row, ex)
    return touch_data

def calibrate(video_path, touch_path):
    # Load the touch data (we use the first 4 points for calibration)
    touches = load_touch_data(touch_path)
    if len(touches) < 4:
        print("Not enough touch data points. At least 4 are required.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return None

    ret, frame = cap.read()
    if not ret:
        print("Could not read a frame from the video.")
        cap.release()
        return None

    # For calibration we define four "raw" points near the corners of the frame.
    h, w = frame.shape[:2]
    raw_points = np.array([[100, 100],
                           [w - 100, 100],
                           [w - 100, h - 100],
                           [100, h - 100]], dtype=np.float32)

    # Use the first four touch events as the target screen coordinates.
    touch_points = np.array([[touches[i]["x"], touches[i]["y"]] for i in range(4)], dtype=np.float32)

    H, status = cv2.findHomography(raw_points, touch_points)
    cap.release()
    return H

if __name__ == "__main__":
    # Define folder paths
    video_folder = os.path.join("videos", "input")
    calibration_folder = os.path.join("data", "trained_model")
    os.makedirs(calibration_folder, exist_ok=True)

    # For example, process video "1.mp4" with its click data "1.txt"
    video_file = os.path.join(video_folder, "1.mp4")
    touch_file = os.path.join(video_folder, "1.txt")
    base_name = os.path.splitext("1.mp4")[0]  # yields "1"

    print(f"Calibrating {base_name}.mp4 using {base_name}.txt...")
    H = calibrate(video_file, touch_file)
    if H is not None:
        # Save the calibration matrix and YAML using the video base name
        calib_npy = os.path.join(calibration_folder, f"calibration_matrix_{base_name}.npy")
        np.save(calib_npy, H)
        calib_yaml = os.path.join(calibration_folder, f"calibration_{base_name}.yaml")
        with open(calib_yaml, "w") as f:
            yaml.dump({"calibration_matrix": H.tolist()}, f)
        print("Calibration successful. Matrix saved to:")
        print("  ", calib_npy)
        print("  ", calib_yaml)
    else:
        print("Calibration failed.")
