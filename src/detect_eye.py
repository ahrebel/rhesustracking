# src/detect_eye.py
import cv2
import deeplabcut
import numpy as np
import os
import pandas as pd
from head_pose_estimator import estimate_head_roll

# Path to your DeepLabCut project config file.
PROJECT_CONFIG = '/Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml'

def detect_eye_and_head(frame, project_config=PROJECT_CONFIG):
    """
    Processes a video frame, runs DLC to detect landmarks, and computes:
      - The average eye coordinate (from left and right pupil)
      - The head roll angle (using left and right eye corners)
      - Returns all landmark positions.
    
    Returns:
      dict: {
         'eye_coord': (x, y),
         'roll_angle': <degrees>,
         'landmarks': {
             'left_pupil': (x, y),
             'right_pupil': (x, y),
             'corner_left': (x, y),
             'corner_right': (x, y)
         }
      }
    """
    # Write the frame to a temporary video file
    temp_video_path = 'temp_frame.mp4'
    height, width = frame.shape[:2]
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
    out.write(frame)
    out.release()
    
    # Run DLC analysis on the temporary video.
    deeplabcut.analyze_videos(project_config, [temp_video_path],
                               save_as_csv=True, destfolder='temp_dlc_results',
                               videotype='.mp4')
    
    # Find the CSV file output by DLC.
    csv_files = [f for f in os.listdir('temp_dlc_results') if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No DLC results found.")
    result_csv = os.path.join('temp_dlc_results', csv_files[0])
    # Read the CSV; note that DLC outputs multi-level headers.
    df = pd.read_csv(result_csv, header=[1, 2])
    
    # Extract landmarks. (Make sure your DLC labels match these names.)
    left_pupil   = (df[('left_pupil','x')].iloc[0],   df[('left_pupil','y')].iloc[0])
    right_pupil  = (df[('right_pupil','x')].iloc[0],  df[('right_pupil','y')].iloc[0])
    corner_left  = (df[('corner_left','x')].iloc[0],  df[('corner_left','y')].iloc[0])
    corner_right = (df[('corner_right','x')].iloc[0], df[('corner_right','y')].iloc[0])
    
    # Compute the eye coordinate as the average of the two pupil positions.
    eye_coord = ((left_pupil[0] + right_pupil[0]) / 2,
                 (left_pupil[1] + right_pupil[1]) / 2)
    
    # Estimate head roll using the eye corner labels.
    roll_angle = estimate_head_roll(corner_left, corner_right)
    
    # Clean up temporary files and folder.
    os.remove(temp_video_path)
    os.remove(result_csv)
    os.rmdir('temp_dlc_results')
    
    return {
        'eye_coord': eye_coord,
        'roll_angle': roll_angle,
        'landmarks': {
            'left_pupil': left_pupil,
            'right_pupil': right_pupil,
            'corner_left': corner_left,
            'corner_right': corner_right
        }
    }

if __name__ == '__main__':
    # For testing: load a sample frame.
    test_frame = cv2.imread("sample_frame.jpg")
    if test_frame is None:
        raise ValueError("sample_frame.jpg not found.")
    result = detect_eye_and_head(test_frame)
    print("Detected eye coordinate:", result['eye_coord'])
    print("Estimated head roll angle (degrees):", result['roll_angle'])
