This script uses DeepPoseKit to perform pose estimation on an input video.
Usage:
    python deep_pose_estimation.py --video path/to/video.mp4 --model path/to/model.h5 --output predictions.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
from deepposekit.io import VideoReader
from deepposekit.models import load_model

def run_pose_estimation(video_path, model_path, output_path):
    # Load the pre-trained DeepPoseKit model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Open the video file using DeepPoseKit's VideoReader
    print(f"Loading video from {video_path}...")
    video = VideoReader(video_path)
    
    # Run predictions (returns an array with shape: [frames, keypoints, 2])
    print("Predicting keypoints...")
    predictions = model.predict(video)
    
    num_frames, num_keypoints, _ = predictions.shape
    print(f"Predicted {num_keypoints} keypoints for {num_frames} frames.")
    
    # Convert predictions to a pandas DataFrame for easy viewing/export
    data = []
    for i in range(num_frames):
        frame_data = {"frame": i}
        for j in range(num_keypoints):
            frame_data[f"x{j}"] = predictions[i, j, 0]
            frame_data[f"y{j}"] = predictions[i, j, 1]
        data.append(frame_data)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run pose estimation with DeepPoseKit")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained DeepPoseKit model (h5 file)")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file for predictions")
    args = parser.parse_args()
    
    run_pose_estimation(args.video, args.model, args.output)
