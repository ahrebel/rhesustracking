import os
import cv2
import numpy as np
import yaml
import pandas as pd
import json
from utils.config_loader import load_yaml_config
from utils.gaze_mapping import map_gaze_to_section

def load_calibration_matrix(matrix_path="data/trained_model/calibration_matrix.npy"):
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Calibration matrix not found at {matrix_path}. Please run calibrate.py first.")
    return np.load(matrix_path)

def get_eye_coordinates(frame):
    """
    Placeholder function to obtain eye coordinates from a video frame.
    Replace this with your DeepLabCut inference call to get the actual eye positions.
    For now, it returns the center of the frame as dummy coordinates.
    """
    height, width = frame.shape[:2]
    return (width / 2, height / 2)

def apply_calibration(raw_point, calib_matrix):
    """
    Apply the calibration transform to raw eye coordinates.
    raw_point: tuple (x, y)
    calib_matrix: 3x3 perspective transformation matrix
    Returns calibrated (x, y) coordinates.
    """
    point = np.array([[raw_point]], dtype=np.float32)
    calibrated_point = cv2.perspectiveTransform(point, calib_matrix)
    return calibrated_point[0][0]

def process_video(video_path, calib_matrix, screen_config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps if fps > 0 else 1/30  # default to 30 fps if unknown

    total_sections = screen_config["rows"] * screen_config["cols"]
    section_times = {section_id: 0.0 for section_id in range(1, total_sections+1)}
    gaze_time_series = []  # Record (timestamp, section_id) pairs
    current_time = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        raw_coords = get_eye_coordinates(frame)
        if raw_coords is None:
            current_time += frame_duration
            continue

        calibrated_coords = apply_calibration(raw_coords, calib_matrix)
        section_id = map_gaze_to_section(calibrated_coords, screen_config)
        section_times[section_id] += frame_duration
        gaze_time_series.append((current_time, section_id))
        current_time += frame_duration

    cap.release()
    print(f"Processed {frame_count} frames from {video_path}")
    return section_times, gaze_time_series

def save_results(video_name, section_times, output_dir="output/gaze_data"):
    os.makedirs(output_dir, exist_ok=True)
    # Save CSV
    csv_path = os.path.join(output_dir, f"{video_name}_gaze.csv")
    df = pd.DataFrame(list(section_times.items()), columns=["SectionID", "TimeSpent"])
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to {csv_path}")
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{video_name}_gaze.json")
    with open(json_path, "w") as f:
        json.dump(section_times, f, indent=4)
    print(f"Saved JSON results to {json_path}")

def load_touch_file(video_name, input_dir="videos/input"):
    # Look for a touch file matching the video name (e.g., trial1_touch.csv or .txt)
    possible_names = [f"{video_name}_touch.csv", f"{video_name}_touch.txt"]
    for name in possible_names:
        path = os.path.join(input_dir, name)
        if os.path.exists(path):
            return path
    return None

def main():
    calib_matrix = load_calibration_matrix()
    screen_config = load_yaml_config("config/screen_config.yaml")
    input_dir = "videos/input"
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov"))]
    if not video_files:
        print("No video files found in videos/input/")
        return
    for video in video_files:
        video_path = os.path.join(input_dir, video)
        section_times, gaze_time_series = process_video(video_path, calib_matrix, screen_config)
        if section_times is not None:
            video_name, _ = os.path.splitext(video)
            save_results(video_name, section_times)
            
            # Process touch events if a corresponding file is found
            touch_file = load_touch_file(video_name, input_dir)
            if touch_file:
                from utils.touch_processing import load_touch_events, correlate_touch_gaze
                touch_df = load_touch_events(touch_file)
                correlation_results = correlate_touch_gaze(gaze_time_series, touch_df, screen_config=screen_config)
                correlation_path = os.path.join("output/gaze_data", f"{video_name}_touch_correlation.json")
                with open(correlation_path, "w") as f:
                    json.dump(correlation_results, f, indent=4)
                print(f"Saved touch correlation results to {correlation_path}")

if __name__ == "__main__":
    main()
