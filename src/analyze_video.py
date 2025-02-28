# src/analyze_video.py
import cv2
import os
import pandas as pd
from detect_eye import detect_eye_and_head

def analyze_video(video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame using the fused detector.
        detection = detect_eye_and_head(frame)
        if detection:
            eye_x, eye_y = detection["eye_coord"]
            roll_angle = detection["roll_angle"]
        else:
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
    parser = argparse.ArgumentParser(description="Analyze video for fused eye tracking using DLC and SLEAP.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()
    
    analyze_video(args.video, args.output)
