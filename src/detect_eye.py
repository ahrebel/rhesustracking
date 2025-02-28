# detect_eye.py
import cv2
import numpy as np
from kalman_filter import KalmanFilter2D
from head_pose_estimator import estimate_head_pose

# Initialize a global Kalman filter for smoothing the eye coordinates.
eye_kalman = KalmanFilter2D()

def run_deeplabcut_analysis(frame):
    """
    Dummy function to simulate DeepLabCut analysis.
    In practice, replace this with your actual DLC code that extracts landmarks.
    Expected to return a dictionary with keys: 'left_pupil', 'right_pupil', 'corner_left', 'corner_right'.
    """
    # For example purposes, these values are hard-coded.
    landmarks = {
        "left_pupil": (100, 150),
        "right_pupil": (130, 150),
        "corner_left": (90, 145),
        "corner_right": (140, 145)
    }
    return landmarks

def detect_eye_and_head(frame):
    """
    Processes a single frame to detect the eye position and estimate head pose.
    
    Returns:
        A dictionary containing:
          - 'eye_coord': Smoothed eye coordinate (tuple).
          - 'landmarks': Raw landmark coordinates from DLC.
          - 'head_pose': Dictionary with 'rotation_vector' and 'translation_vector'.
    """
    # Obtain landmarks from DeepLabCut analysis.
    landmarks = run_deeplabcut_analysis(frame)
    
    # Calculate raw eye coordinate as the average of left and right pupil positions.
    raw_eye_x = (landmarks["left_pupil"][0] + landmarks["right_pupil"][0]) / 2
    raw_eye_y = (landmarks["left_pupil"][1] + landmarks["right_pupil"][1]) / 2
    raw_eye_coord = (raw_eye_x, raw_eye_y)
    
    # Apply Kalman filter to smooth the eye coordinate.
    smoothed_eye_coord = eye_kalman.update(raw_eye_coord)
    
    # Define a camera matrix (example values); you may replace these with calibrated parameters.
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    
    # Estimate head pose based on the landmarks.
    head_pose = estimate_head_pose(landmarks, camera_matrix)
    
    return {
        "eye_coord": smoothed_eye_coord,
        "landmarks": landmarks,
        "head_pose": head_pose
    }
