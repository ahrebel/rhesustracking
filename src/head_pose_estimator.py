# head_pose_estimator.py
import numpy as np

def estimate_head_roll(left_corner, right_corner):
    """
    Estimate the head roll angle (in degrees) based on the positions of the left and right eye corners.
    A roll angle of 0° indicates that the corners are horizontally aligned.
    
    Parameters:
      left_corner (tuple): (x, y) coordinate of the left eye corner.
      right_corner (tuple): (x, y) coordinate of the right eye corner.
    
    Returns:
      float: Roll angle in degrees.
    """
    dx = right_corner[0] - left_corner[0]
    dy = right_corner[1] - left_corner[1]
    angle_rad = np.arctan2(dy, dx)
    return np.degrees(angle_rad)
