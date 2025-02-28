# src/detect_eye.py
import cv2
from detect_eye_dlc import detect_eye_and_head_dlc
from detect_eye_sleap import detect_eye_and_head_sleap
from fuse_landmarks import fuse_results

def detect_eye_and_head(frame, project_config=None):
    """
    Processes a video frame by running both DLC and SLEAP detection,
    then fuses the results.
    
    Returns:
      dict: {
          'eye_coord': fused (x, y),
          'roll_angle': fused or DLC roll angle,
          'landmarks': {
              'DLC': <DLC landmarks>,
              'SLEAP': <SLEAP landmarks>
          }
      }
    """
    # If project_config is not provided, use the default from DLC module.
    if project_config is None:
        from detect_eye_dlc import PROJECT_CONFIG
        project_config = PROJECT_CONFIG
    
    dlc_result = detect_eye_and_head_dlc(frame, project_config)
    sleap_result = detect_eye_and_head_sleap(frame)
    fused = fuse_results(dlc_result, sleap_result)
    return fused
