import cv2
import deeplabcut
import numpy as np
import os
import pandas as pd

# Path to your DeepLabCut project config file.
# Update this path to point to your DLC project configuration.
PROJECT_CONFIG = 'path/to/config.yaml'

def detect_eye(frame, project_config=PROJECT_CONFIG, keypoint='eye'):
    """
    Given a video frame, detect the eye landmark using a trained DLC model.
    Returns (x, y) coordinates for the detected eye.
    """
    # Save the current frame to a temporary video file.
    temp_video_path = 'temp_frame.mp4'
    height, width = frame.shape[:2]
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
    out.write(frame)
    out.release()
    
    # Run DLC analysis on the temporary video.
    # This will create a CSV file with predictions.
    deeplabcut.analyze_videos(project_config, [temp_video_path],
                               save_as_csv=True, destfolder='temp_dlc_results',
                               videotype='.mp4')
    
    # Find the CSV file output by DLC.
    csv_files = [f for f in os.listdir('temp_dlc_results') if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No DLC results found.")
    result_csv = os.path.join('temp_dlc_results', csv_files[0])
    df = pd.read_csv(result_csv, header=[1, 2])
    
    # Assume the keypoint is labeled with the given name.
    # Extract the x and y coordinates from the first (and only) frame.
    x = df[(keypoint, 'x')].iloc[0]
    y = df[(keypoint, 'y')].iloc[0]
    
    # Clean up temporary files.
    os.remove(temp_video_path)
    os.remove(result_csv)
    os.rmdir('temp_dlc_results')
    
    return (x, y)

if __name__ == "__main__":
    # For testing, load a sample frame.
    test_frame = cv2.imread("sample_frame.jpg")
    if test_frame is None:
        raise ValueError("sample_frame.jpg not found.")
    eye_coord = detect_eye(test_frame)
    print("Detected eye coordinate:", eye_coord)
