# src/detect_eye_sleap.py
import cv2
import numpy as np
import sleap

# Adjust the model_path to your trained SLEAP model checkpoint.
MODEL_PATH = 'models/sleap_model.ckpt'
model = sleap.load_model(MODEL_PATH)

def detect_eye_and_head_sleap(frame):
    """
    Processes a video frame using SLEAP to detect eye landmarks.
    
    Returns:
        dict: {
          'eye_coord': (x, y),   # Average of left and right eye positions
          'landmarks': {         # Raw landmark positions from SLEAP
              'left_eye': (x, y),
              'right_eye': (x, y)
          },
          'roll_angle': None     # SLEAP does not compute head roll (optional)
        }
    """
    # Convert frame to RGB (SLEAP expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = model.predict(frame_rgb)
    
    if predictions and predictions[0].instances:
        instance = predictions[0].instances[0]
        landmarks = {}
        for kp in instance.keypoints:
            if kp.label == "left_eye":
                landmarks["left_eye"] = (kp.x, kp.y)
            elif kp.label == "right_eye":
                landmarks["right_eye"] = (kp.x, kp.y)
        if "left_eye" in landmarks and "right_eye" in landmarks:
            eye_coord = (
                (landmarks["left_eye"][0] + landmarks["right_eye"][0]) / 2,
                (landmarks["left_eye"][1] + landmarks["right_eye"][1]) / 2
            )
            return {
                "eye_coord": eye_coord,
                "landmarks": landmarks,
                "roll_angle": None  # Roll angle is not computed via SLEAP
            }
    return None
