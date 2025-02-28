# detect_eye.py
import cv2
import deeplabcut
import numpy as np
import os
import pandas as pd
from head_pose_estimator import estimate_head_pose

# Update this path to point to your DLC project config file.
PROJECT_CONFIG = '/Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml'

def detect_eye_and_head(frame, project_config=PROJECT_CONFIG):
    """
    Detect eye landmarks and estimate head pose for a given frame.
    
    Returns a dictionary with:
      - 'eye': a tuple (x, y) representing the eye coordinate (e.g., the average of the left and right pupil)
      - 'landmarks': a dict with keys 'left_pupil', 'right_pupil', 'corner_left', 'corner_right'
      - 'head_pose': a dict with keys 'rvec' and 'tvec' (or None if estimation fails)
    """
    # Save the current frame to a temporary video file.
    temp_video_path = 'temp_frame.mp4'
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, 1, (width, height))
    out.write(frame)
    out.release()
    
    # Run DLC analysis on the temporary video.
    deeplabcut.analyze_videos(project_config, [temp_video_path],
                               save_as_csv=True, destfolder='temp_dlc_results',
                               videotype='.mp4')
    
    # Look for the CSV output.
    csv_files = [f for f in os.listdir('temp_dlc_results') if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No DLC results found.")
    result_csv = os.path.join('temp_dlc_results', csv_files[0])
    # Read the CSV file (assumes multi-index header)
    df = pd.read_csv(result_csv, header=[1, 2])
    
    # Extract landmarks for the required keypoints.
    landmarks = {}
    for key in ['left_pupil', 'right_pupil', 'corner_left', 'corner_right']:
        landmarks[key] = (df[(key, 'x')].iloc[0], df[(key, 'y')].iloc[0])
    
    # Compute the eye coordinate (e.g., the average of left and right pupil positions).
    eye_x = (landmarks['left_pupil'][0] + landmarks['right_pupil'][0]) / 2.0
    eye_y = (landmarks['left_pupil'][1] + landmarks['right_pupil'][1]) / 2.0
    eye_coord = (eye_x, eye_y)
    
    # Define a dummy camera matrix based on frame dimensions.
    focal_length = width  # approximate focal length in pixels
    center = (width / 2, height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    
    success, rvec, tvec = estimate_head_pose(landmarks, camera_matrix)
    head_pose = {'rvec': rvec, 'tvec': tvec} if success else None
    
    # Cleanup temporary files.
    os.remove(temp_video_path)
    os.remove(result_csv)
    os.rmdir('temp_dlc_results')
    
    return {'eye': eye_coord, 'landmarks': landmarks, 'head_pose': head_pose}

if __name__ == "__main__":
    test_frame = cv2.imread("sample_frame.jpg")
    if test_frame is None:
        raise ValueError("sample_frame.jpg not found.")
    result = detect_eye_and_head(test_frame)
    print("Detected result:", result)
