# src/analyze_video.py
import cv2
import numpy as np
import os
import glob
import pandas as pd
from detect_eye import detect_eye_and_head

def load_calibration_matrix(calibration_file='data/trained_model/calibration_matrix_1.npy'):
    return np.load(calibration_file)

def analyze_video(video_path, calibration_matrix_file='data/trained_model/calibration_matrix_1.npy'):
    """
    Processes a video frame-by-frame, extracts eye landmarks and head roll,
    applies calibration to map the raw eye coordinates to screen coordinates,
    and saves the output including head–pose information.
    """
    # Load the calibration matrix.
    H = load_calibration_matrix(calibration_matrix_file)
    
    cap = cv2.VideoCapture(video_path)
    gaze_data = []
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            result = detect_eye_and_head(frame)
        except Exception as e:
            print("Error during eye detection:", e)
            continue
        
        # Convert detected eye coordinate to homogeneous coordinates.
        pt = np.array([result['eye_coord'][0], result['eye_coord'][1], 1]).reshape(3, 1)
        mapped = H.dot(pt)
        mapped /= mapped[2, 0]  # Normalize
        
        gaze_data.append({
            'frame': frame_number,
            'raw_eye_x': result['eye_coord'][0],
            'raw_eye_y': result['eye_coord'][1],
            'screen_x': mapped[0, 0],
            'screen_y': mapped[1, 0],
            'head_roll': result['roll_angle']
        })
        
        frame_number += 1
    
    cap.release()
    
    # Save gaze data to CSV.
    df = pd.DataFrame(gaze_data)
    output_file = os.path.join('data/analysis_output', os.path.basename(video_path).split('.')[0] + '_gaze.csv')
    os.makedirs('data/analysis_output', exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Gaze analysis saved to:", output_file)

if __name__ == "__main__":
    video_files = glob.glob('videos/input/*.mp4')
    for video in video_files:
        analyze_video(video)
