# src/analyze_video.py
import cv2
import numpy as np
import os
import glob
import pandas as pd
import pickle
from detect_eye import detect_eye_and_head

def load_calibration_matrix(calibration_file='data/trained_model/calibration_matrix_1.npy'):
    return np.load(calibration_file)

def load_gaze_mapping_model(model_path='data/trained_model/gaze_mapping_model.pkl'):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    return None

def analyze_video(video_path, calibration_matrix_file='data/trained_model/calibration_matrix_1.npy', mapping_model_path='data/trained_model/gaze_mapping_model.pkl'):
    """
    Processes the video frame-by-frame:
      - Extracts eye landmarks and head roll using DLC.
      - Retrieves the timestamp for each frame.
      - Maps raw eye coordinates (and head roll) to screen coordinates using either:
            a) a trained regression model (if available), or
            b) a pre–computed homography matrix.
      - Saves a CSV file with frame, timestamp, raw eye coordinates, head roll, and mapped screen coordinates.
    """
    mapping_model = load_gaze_mapping_model(mapping_model_path)
    if mapping_model is None:
        H = load_calibration_matrix(calibration_matrix_file)
    
    cap = cv2.VideoCapture(video_path)
    gaze_data = []
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Get timestamp in seconds.
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        try:
            result = detect_eye_and_head(frame)
        except Exception as e:
            print("Error during eye detection:", e)
            continue
        
        raw_eye_x, raw_eye_y = result['eye_coord']
        head_roll = result['roll_angle']
        
        if mapping_model:
            # Use the regression model.
            features = np.array([[raw_eye_x, raw_eye_y, head_roll]])
            screen_coords = mapping_model.predict(features)[0]
            screen_x, screen_y = screen_coords
        else:
            # Use homography-based calibration.
            pt = np.array([raw_eye_x, raw_eye_y, 1]).reshape(3, 1)
            mapped = H.dot(pt)
            mapped /= mapped[2, 0]
            screen_x, screen_y = mapped[0, 0], mapped[1, 0]
        
        gaze_data.append({
            'frame': frame_number,
            'timestamp_sec': timestamp,
            'raw_eye_x': raw_eye_x,
            'raw_eye_y': raw_eye_y,
            'head_roll': head_roll,
            'screen_x': screen_x,
            'screen_y': screen_y
        })
        frame_number += 1
    
    cap.release()
    
    df = pd.DataFrame(gaze_data)
    output_file = os.path.join('data/analysis_output', os.path.basename(video_path).split('.')[0] + '_gaze.csv')
    os.makedirs('data/analysis_output', exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Gaze analysis saved to:", output_file)

if __name__ == "__main__":
    video_files = glob.glob('videos/input/*.mp4')
    for video in video_files:
        analyze_video(video)
