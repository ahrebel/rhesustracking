#!/usr/bin/env python
import cv2
import os
import pandas as pd
import warnings
from detect_eye import detect_eye_and_landmarks
warnings.filterwarnings(
    "ignore",
    message="`layer.apply` is deprecated and will be removed in a future version."
)

def process_video(video_path, config_path, output_csv_path):
    """
    Process the input video to extract eye landmarks using DeepLabCut.
    Saves a CSV with one row per frame containing:
      frame, time, left/right pupil (x,y), left/right corner (x,y), roll_angle.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            detection = detect_eye_and_landmarks(frame, config_path=config_path)
            landmarks = detection['landmarks']
            roll_angle = detection.get('roll_angle', None)
        except Exception as e:
            print(f"Frame {frame_num}: Detection error: {e}")
            landmarks = {
                'left_pupil': (None, None),
                'right_pupil': (None, None),
                'corner_left': (None, None),
                'corner_right': (None, None)
            }
            roll_angle = None
        
        timestamp = frame_num / fps if fps > 0 else None
        results.append({
            "frame": frame_num,
            "time": timestamp,
            "left_pupil_x": landmarks['left_pupil'][0],
            "left_pupil_y": landmarks['left_pupil'][1],
            "right_pupil_x": landmarks['right_pupil'][0],
            "right_pupil_y": landmarks['right_pupil'][1],
            "corner_left_x": landmarks['corner_left'][0],
            "corner_left_y": landmarks['corner_left'][1],
            "corner_right_x": landmarks['corner_right'][0],
            "corner_right_y": landmarks['corner_right'][1],
            "roll_angle": roll_angle
        })
        frame_num += 1
    
    cap.release()
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Video processing complete. Results saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Process a video to extract eye landmarks using DeepLabCut"
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--config", required=True, help="Path to the DLC config.yaml file")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    process_video(args.video, args.config, args.output)
