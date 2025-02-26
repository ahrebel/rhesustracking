import cv2
import numpy as np
import os
import glob
import pandas as pd
from detect_eye import detect_eye

def load_calibration_matrix(calibration_file='data/trained_model/calibration_matrix_1.npy'):
    return np.load(calibration_file)

def analyze_video(video_path, calibration_matrix_file='data/trained_model/calibration_matrix_1.npy'):
    """
    Process the video frame-by-frame, detect the eye coordinates, apply calibration,
    and save the mapped gaze points to a CSV file.
    """
    # Load the calibration matrix.
    H = load_calibration_matrix(calibration_matrix_file)
    
    cap = cv2.VideoCapture(video_path)
    gaze_points = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Detect the eye landmark using DLC.
            eye_coord = detect_eye(frame)
        except Exception as e:
            print("Error during eye detection:", e)
            continue
        
        # Convert the detected point to homogeneous coordinates.
        pt = np.array([eye_coord[0], eye_coord[1], 1]).reshape(3, 1)
        mapped = H.dot(pt)
        mapped /= mapped[2, 0]  # Normalize
        gaze_points.append((mapped[0, 0], mapped[1, 0]))
    
    cap.release()
    
    # Save gaze points to CSV.
    df = pd.DataFrame(gaze_points, columns=['x', 'y'])
    output_file = os.path.join('data/analysis_output', os.path.basename(video_path).split('.')[0] + '_gaze.csv')
    os.makedirs('data/analysis_output', exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Gaze analysis saved to:", output_file)

if __name__ == "__main__":
    # Process all videos in the input directory.
    video_files = glob.glob('videos/input/*.mp4')
    for video in video_files:
        analyze_video(video)
