import os
import cv2
import numpy as np
import yaml
import csv
import json
import logging
import pandas as pd
from utils.config_loader import load_yaml_config
from utils.gaze_mapping import map_gaze_to_section

def load_calibration_matrix(video_base, calib_dir="data/trained_model"):
    calib_path = os.path.join(calib_dir, f"calibration_matrix_{video_base}.npy")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration matrix not found at {calib_path}. Please run calibrate.py for video {video_base}.")
    return np.load(calib_path)

def get_eye_coordinates(frame):
    """
    Placeholder: returns the center of the frame.
    Replace with your DeepLabCut or other eye tracking method.
    """
    height, width = frame.shape[:2]
    return (width / 2, height / 2)

def apply_calibration(raw_point, calib_matrix):
    """
    Apply a homography transform to raw (x,y) coordinates.
    """
    pt = np.array([raw_point[0], raw_point[1], 1.0])
    transformed = calib_matrix.dot(pt)
    if transformed[2] != 0:
        transformed /= transformed[2]
    return (float(transformed[0]), float(transformed[1]))

def process_video(video_path, calib_matrix, screen_config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps if fps > 0 else 1/30
    total_sections = screen_config["rows"] * screen_config["cols"]
    section_times = {section_id: 0.0 for section_id in range(1, total_sections+1)}
    gaze_time_series = []
    current_time = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        raw_coords = get_eye_coordinates(frame)
        # If gaze data is missing, skip this frame.
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

def save_results(video_base, section_times, output_dir="output/gaze_data"):
    os.makedirs(output_dir, exist_ok=True)
    # Save CSV
    csv_path = os.path.join(output_dir, f"{video_base}_gaze.csv")
    df = pd.DataFrame(list(section_times.items()), columns=["SectionID", "TimeSpent"])
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to {csv_path}")
    
    # Save JSON (combined data can be expanded later)
    json_path = os.path.join(output_dir, f"{video_base}_gaze.json")
    with open(json_path, "w") as f:
        json.dump(section_times, f, indent=4)
    print(f"Saved JSON results to {json_path}")

def load_touch_file(video_base, input_dir="videos/input"):
    possible_names = [f"{video_base}_touch.csv", f"{video_base}_touch.txt", f"{video_base}_touch_events.csv"]
    for name in possible_names:
        path = os.path.join(input_dir, name)
        if os.path.exists(path):
            return path
    return None

def load_touch_events(file_path):
    touch_events = []
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # For analysis, we assume the 'timestamp' column is in seconds (float)
                t = float(row["timestamp"])
                x = float(row["x"])
                y = float(row["y"])
                touch_events.append({"time": t, "x": x, "y": y})
            except Exception as e:
                print("Error parsing touch row:", row, e)
    return touch_events

def correlate_touch_gaze(gaze_time_series, touch_events):
    correlation = []
    for touch in touch_events:
        t_time = touch["time"]
        # Find the gaze point nearest in time to the touch event
        closest = min(gaze_time_series, key=lambda g: abs(g[0] - t_time))
        correlation.append({
            "touch_time": t_time,
            "gaze_time": closest[0],
            "gaze_section": closest[1],
            "time_diff": abs(closest[0] - t_time)
        })
    return correlation

def main():
    input_dir = "videos/input"
    calib_dir = "data/trained_model"
    screen_config = load_yaml_config("config/screen_config.yaml")
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov"))]
    if not video_files:
        print("No video files found in", input_dir)
        return

    for video in video_files:
        video_path = os.path.join(input_dir, video)
        video_base, _ = os.path.splitext(video)
        try:
            calib_matrix = load_calibration_matrix(video_base, calib_dir)
        except Exception as e:
            print(e)
            continue

        section_times, gaze_time_series = process_video(video_path, calib_matrix, screen_config)
        if section_times is None:
            continue

        save_results(video_base, section_times)

        touch_file = load_touch_file(video_base, input_dir)
        if touch_file:
            touch_events = load_touch_events(touch_file)
            if touch_events:
                correlation = correlate_touch_gaze(gaze_time_series, touch_events)
                correlation_path = os.path.join("output/gaze_data", f"{video_base}_touch_correlation.json")
                with open(correlation_path, "w") as f:
                    json.dump(correlation, f, indent=4)
                print(f"Saved touch correlation results to {correlation_path}")
            else:
                print(f"No valid touch events found in {touch_file}")
        else:
            print(f"No touch file found for video {video_base}")

if __name__ == "__main__":
    main()
