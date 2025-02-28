# head_pose_estimator.py
import numpy as np

def estimate_head_roll(left_corner, right_corner):
    """
    Estimate the head roll angle (rotation about the longitudinal axis) based on the 
    positions of the left and right eye corners.
    
    Parameters:
      left_corner (tuple): (x, y) coordinates of the left eye corner.
      right_corner (tuple): (x, y) coordinates of the right eye corner.
    
    Returns:
      float: Roll angle in degrees. 0 means the corners are level.
    """
    dx = right_corner[0] - left_corner[0]
    dy = right_corner[1] - left_corner[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg
