# analyze_video.py
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
        
        # Process the frame for eye detection and head pose estimation.
        detection = detect_eye_and_head(frame)
        eye_x, eye_y = detection["eye_coord"]
        rotation_vector = detection["head_pose"]["rotation_vector"]
        translation_vector = detection["head_pose"]["translation_vector"]
        
        results.append({
            "frame": frame_num,
            "eye_x": eye_x,
            "eye_y": eye_y,
            "rotation_vector_x": rotation_vector[0][0],
            "rotation_vector_y": rotation_vector[1][0],
            "rotation_vector_z": rotation_vector[2][0],
            "translation_vector_x": translation_vector[0][0],
            "translation_vector_y": translation_vector[1][0],
            "translation_vector_z": translation_vector[2][0]
        })
        
        frame_num += 1
    
    cap.release()
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Analysis complete. Results saved to {output_csv_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze video for eye tracking.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()
    
    analyze_video(args.video, args.output)
