import cv2
import os
import pandas as pd
import tempfile
from head_pose_estimator import estimate_head_roll
import deeplabcut  # Ensure DLC is installed

# Path to your DeepLabCut project config file.
PROJECT_CONFIG = '/Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml'

def detect_eye_and_head_dlc(frame, project_config=PROJECT_CONFIG):
    """
    Process a single video frame using DeepLabCut to detect landmarks and compute:
      - The average eye coordinate (from left/right pupil)
      - The head roll angle (using left/right eye corners)
    
    Returns a dictionary:
      {
        'eye_coord': (x, y),
        'roll_angle': roll_angle,
        'landmarks': {
            'left_pupil': (x, y),
            'right_pupil': (x, y),
            'corner_left': (x, y),
            'corner_right': (x, y)
        }
      }
    """
    height, width = frame.shape[:2]
    
    # Use a temporary directory for robust temporary file management.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, 'temp_frame.mp4')
        # Write the single frame to a video file.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 1, (width, height))
        out.write(frame)
        out.release()
        
        # Run DLC analysis on the temporary video.
        deeplabcut.analyze_videos(project_config, [temp_video_path],
                                   save_as_csv=True, destfolder=temp_dir,
                                   videotype='.mp4')
        
        # Find the output CSV produced by DLC.
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No DLC results found. Check that the video was analyzed correctly.")
        result_csv = os.path.join(temp_dir, csv_files[0])
        
        # DLC outputs a CSV with multi-level column headers.
        df = pd.read_csv(result_csv, header=[1, 2])
        
        # Extract landmarks from the first frame (adjust label names as needed).
        try:
            left_pupil   = (df[('left_pupil', 'x')].iloc[0],   df[('left_pupil', 'y')].iloc[0])
            right_pupil  = (df[('right_pupil', 'x')].iloc[0],  df[('right_pupil', 'y')].iloc[0])
            corner_left  = (df[('corner_left', 'x')].iloc[0],  df[('corner_left', 'y')].iloc[0])
            corner_right = (df[('corner_right', 'x')].iloc[0], df[('corner_right', 'y')].iloc[0])
        except KeyError as e:
            raise ValueError("Expected landmark label not found in DLC output CSV.") from e
        
        # Compute the average eye coordinate.
        eye_coord = ((left_pupil[0] + right_pupil[0]) / 2,
                     (left_pupil[1] + right_pupil[1]) / 2)
        
        # Compute the head roll angle.
        roll_angle = estimate_head_roll(corner_left, corner_right)
    
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect eye and head landmarks on a single frame using DeepLabCut")
    parser.add_argument("--image", required=True, help="Path to an image file (a single frame)")
    parser.add_argument("--config", default=PROJECT_CONFIG, help="Path to DLC config.yaml")
    args = parser.parse_args()
    
    # Read the image
    frame = cv2.imread(args.image)
    if frame is None:
        raise ValueError(f"Could not load image: {args.image}")
    
    detection = detect_eye_and_head_dlc(frame, project_config=args.config)
    print("Detection result:")
    print(detection)
