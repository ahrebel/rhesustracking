import cv2
import deeplabcut
import numpy as np
import os
import pandas as pd
import tempfile
import glob

# Path to your DeepLabCut project config file.
PROJECT_CONFIG = '/Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml'

def detect_eye(frame, project_config=PROJECT_CONFIG, keypoint='left_pupil'):
    """
    Given a video frame, detect the specified keypoint (default: 'left_pupil')
    using a trained DLC model. Returns (x, y) coordinates for the detected landmark.
    """
    height, width = frame.shape[:2]
    
    # Write the frame to a temporary video file.
    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_video_fd)  # Close the file descriptor; we will write using OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, 1, (width, height))
    out.write(frame)
    out.release()
    
    # Create a temporary directory for DLC results.
    with tempfile.TemporaryDirectory() as temp_dlc_dir:
        deeplabcut.analyze_videos(project_config,
                                  [temp_video_path],
                                  save_as_csv=True,
                                  destfolder=temp_dlc_dir,
                                  videotype='.mp4')
        
        # Locate the DLC CSV file.
        csv_files = glob.glob(os.path.join(temp_dlc_dir, '*.csv'))
        if not csv_files:
            os.remove(temp_video_path)
            raise ValueError("No DLC results found in the temporary directory.")
        result_csv = csv_files[0]
        df = pd.read_csv(result_csv, header=[1, 2])
        
        # Check that the expected key is available.
        if (keypoint, 'x') not in df.columns:
            os.remove(temp_video_path)
            available = df.columns.levels[0].tolist() if hasattr(df.columns, "levels") else df.columns.tolist()
            raise KeyError(f"Keypoint '{keypoint}' not found in DLC output. Available keypoints: {available}")
        
        # Extract coordinates from the first frame.
        x = df[(keypoint, 'x')].iloc[0]
        y = df[(keypoint, 'y')].iloc[0]
    
    # Clean up the temporary video file.
    os.remove(temp_video_path)
    
    return (x, y)

if __name__ == "__main__":
    # For testing, load a sample frame.
    test_frame = cv2.imread("sample_frame.jpg")
    if test_frame is None:
        raise ValueError("sample_frame.jpg not found.")
    eye_coord = detect_eye(test_frame)
    print("Detected eye coordinate:", eye_coord)
