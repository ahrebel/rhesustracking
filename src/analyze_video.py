import os
import cv2
import numpy as np
import csv
import json
import argparse

# Import the SLEAP-based eye detection function
from detect_eye import get_eye_coordinates

def load_calibration(video_base):
    """
    Loads the calibration matrix for a given video.
    Expected filename: calibration_matrix_<video_base>.npy in data/trained_model/
    """
    matrix_path = os.path.join("data", "trained_model", f"calibration_matrix_{video_base}.npy")
    if os.path.exists(matrix_path):
        calibration_matrix = np.load(matrix_path)
        return calibration_matrix
    else:
        print(f"[ERROR] Calibration matrix not found for video base '{video_base}'.")
        return None

def map_eye_to_screen(raw_coords, calibration_matrix):
    """
    Maps raw eye coordinates to screen coordinates using the calibration matrix.
    """
    pt = np.array([raw_coords[0], raw_coords[1], 1.0])
    mapped = calibration_matrix.dot(pt)
    mapped /= mapped[2]  # Normalize by the homogeneous coordinate
    return (mapped[0], mapped[1])

def process_video(video_path, calibration_matrix, grid_cols=10, grid_rows=11):
    """
    Process the video:
      - Use SLEAP to get eye coordinates per frame
      - Map coordinates to screen space using the calibration matrix
      - Assign each point to a grid section and sum up fixation times
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    section_width = width / grid_cols
    section_height = height / grid_rows

    # Initialize fixation times dictionary: keys are (col, row)
    fixation_times = {(col, row): 0.0 for col in range(grid_cols) for row in range(grid_rows)}

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # Get raw eye coordinates from the SLEAP model
        raw_coords = get_eye_coordinates(frame)
        # Map them using the calibration matrix
        screen_coords = map_eye_to_screen(raw_coords, calibration_matrix)

        # Determine grid indices
        col = int(screen_coords[0] // section_width)
        row = int(screen_coords[1] // section_height)
        col = max(0, min(col, grid_cols - 1))
        row = max(0, min(row, grid_rows - 1))

        # Accumulate the fixation time for this section (frame duration)
        fixation_times[(col, row)] += 1.0 / fps

    cap.release()
    return fixation_times

def save_results(fixation_times, output_csv, output_json):
    """
    Save the fixation times to CSV and JSON.
    """
    # Save CSV
    with open(output_csv, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["Section", "Fixation_Time_sec"])
        for key, duration in fixation_times.items():
            writer.writerow([f"{key}", duration])
    # Save JSON
    with open(output_json, 'w') as fj:
        json.dump({f"{k}": v for k, v in fixation_times.items()}, fj, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Analyze video for gaze fixation times per screen section.")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--output", default="data/analysis_output/", help="Directory to save analysis results")
    args = parser.parse_args()

    video_path = args.video
    video_base = os.path.splitext(os.path.basename(video_path))[0]

    calibration_matrix = load_calibration(video_base)
    if calibration_matrix is None:
        print("[ERROR] Calibration failed. Exiting.")
        return

    print(f"[INFO] Processing video '{video_path}' with calibration from '{video_base}'...")
    fixation_times = process_video(video_path, calibration_matrix)
    if fixation_times is None:
        print("[ERROR] Video processing failed.")
        return

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, f"{video_base}_fixation_times.csv")
    json_path = os.path.join(args.output, f"{video_base}_fixation_times.json")
    save_results(fixation_times, csv_path, json_path)
    print(f"[INFO] Analysis complete. Results saved to:\n  CSV: {csv_path}\n  JSON: {json_path}")

if __name__ == "__main__":
    main()
