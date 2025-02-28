import cv2
import os
import pandas as pd
from detect_eye import detect_eye_and_head

def analyze_video(video_path, config_path, output_csv_path):
    """
    Analyze a video using DeepLabCut-based detection.
    For each frame, detect landmarks and compute eye coordinate and head roll.
    The raw predictions (per frame) are saved to a CSV.
    
    Args:
        video_path (str): Path to the input video.
        config_path (str): Path to the DLC config.yaml file.
        output_csv_path (str): Path where the CSV with results will be saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_num = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect landmarks on this frame using DLC.
        try:
            detection = detect_eye_and_head(frame, project_config=config_path)
            eye_x, eye_y = detection["eye_coord"]
            roll_angle = detection["roll_angle"]
        except Exception as e:
            print(f"Frame {frame_num}: Detection failed with error: {e}")
            eye_x, eye_y, roll_angle = None, None, None
        
        results.append({
            "frame": frame_num,
            "eye_x": eye_x,
            "eye_y": eye_y,
            "roll_angle": roll_angle
        })
        frame_num += 1
    
    cap.release()
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Analysis complete. Results saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze a video using DeepLabCut for eye and head detection")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--config", required=True, help="Path to the DLC config.yaml file")
    parser.add_argument("--output", required=True, help="Path to the output CSV file")
    args = parser.parse_args()
    
    analyze_video(args.video, args.config, args.output)
