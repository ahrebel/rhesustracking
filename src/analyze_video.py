import cv2
import deeplabcut
import numpy as np
import os
import glob
import pandas as pd

# Import your detect_eye function
from detect_eye import detect_eye

def load_calibration_matrix(calibration_file='data/trained_model/calibration_matrix_1.npy'):
    return np.load(calibration_file)

def analyze_video(video_path, calibration_matrix_file='data/trained_model/calibration_matrix_1.npy'):
    """
    Process the video frame-by-frame, detect the eye coordinates, apply calibration,
    add a timestamp for each frame, and save the (time, x, y) gaze points to a CSV file.
    """
    # Load the calibration matrix.
    H = load_calibration_matrix(calibration_matrix_file)
    
    # Open the video.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video:", video_path)
    
    # Get the frames per second (fps) for timestamp calculation.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # default fallback
    
    gaze_points = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp in seconds.
        timestamp = frame_index / fps

        try:
            # Detect eye using your DLC pipeline.
            eye_coord = detect_eye(frame)
        except Exception as e:
            print("Error during eye detection:", e)
            frame_index += 1
            continue
        
        # Convert eye coordinate to homogeneous coordinates.
        pt = np.array([eye_coord[0], eye_coord[1], 1]).reshape(3, 1)
        mapped = H.dot(pt)
        mapped /= mapped[2, 0]  # Normalize to get calibrated (x, y)

        gaze_points.append((timestamp, mapped[0, 0], mapped[1, 0]))
        frame_index += 1

    cap.release()
    
    # Save the gaze points with timestamps.
    df = pd.DataFrame(gaze_points, columns=['time', 'x', 'y'])
    base = os.path.basename(video_path).split('.')[0]
    output_file = os.path.join('data/analysis_output', f"{base}_gaze.csv")
    os.makedirs('data/analysis_output', exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Gaze analysis saved to:", output_file)

if __name__ == "__main__":
    # Process all MP4 videos in your input folder.
    video_files = glob.glob('videos/input/*.mp4')
    for video in video_files:
        analyze_video(video)
