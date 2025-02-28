# analyze_video.py
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
    Process the video frame-by-frame, detect eye and head pose, apply calibration,
    and save the mapped gaze and pose parameters to a CSV file.
    """
    H = load_calibration_matrix(calibration_matrix_file)
    
    cap = cv2.VideoCapture(video_path)
    records = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        try:
            result = detect_eye_and_head(frame)
            eye_coord = result['eye']
            head_pose = result['head_pose']
        except Exception as e:
            print(f"Error during eye detection at frame {frame_idx}:", e)
            continue
        
        # Map the eye coordinate using the calibration matrix H.
        pt = np.array([eye_coord[0], eye_coord[1], 1]).reshape(3, 1)
        mapped = H.dot(pt)
        mapped /= mapped[2, 0]  # Normalize
        screen_x, screen_y = mapped[0, 0], mapped[1, 0]
        
        # Extract head pose parameters (if available)
        if head_pose is not None:
            rvec = head_pose['rvec'].flatten()
            tvec = head_pose['tvec'].flatten()
        else:
            rvec = [None, None, None]
            tvec = [None, None, None]
        
        records.append({
            'frame': frame_idx,
            'eye_x': eye_coord[0],
            'eye_y': eye_coord[1],
            'screen_x': screen_x,
            'screen_y': screen_y,
            'rvec_x': rvec[0],
            'rvec_y': rvec[1],
            'rvec_z': rvec[2],
            'tvec_x': tvec[0],
            'tvec_y': tvec[1],
            'tvec_z': tvec[2]
        })
    
    cap.release()
    df = pd.DataFrame(records)
    output_file = os.path.join('data/analysis_output', os.path.basename(video_path).split('.')[0] + '_gaze.csv')
    os.makedirs('data/analysis_output', exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Gaze analysis saved to:", output_file)

if __name__ == "__main__":
    # Process all .mp4 videos in the input directory.
    video_files = glob.glob('videos/input/*.mp4')
    for video in video_files:
        analyze_video(video)
