# head_pose_estimator.py
import cv2
import numpy as np

def estimate_head_pose(landmarks, camera_matrix, dist_coeffs=np.zeros((4,1))):
    """
    Estimate head pose using 2D landmarks and predefined 3D model points.
    
    Parameters:
      landmarks: dict with keys: 'left_pupil', 'right_pupil', 'corner_left', 'corner_right'
                 Each value is a tuple (x, y).
      camera_matrix: Intrinsic camera matrix.
      dist_coeffs: Distortion coefficients (default: zeros).
    
    Returns:
      success: Boolean indicating if the pose was estimated.
      rvec: Rotation vector.
      tvec: Translation vector.
      
    Note:
      The model points below are approximate; you may wish to adjust them based on
      your experimental setup.
    """
    # Define approximate 3D model points (in millimeters or arbitrary units)
    model_points = np.array([
        [-30.0, 0.0, 0.0],    # left pupil approximate 3D location
        [30.0, 0.0, 0.0],     # right pupil
        [-40.0, -30.0, 0.0],  # left eye corner
        [40.0, -30.0, 0.0]    # right eye corner
    ], dtype=np.float32)
    
    # Extract corresponding 2D image points
    image_points = np.array([
        landmarks['left_pupil'],
        landmarks['right_pupil'],
        landmarks['corner_left'],
        landmarks['corner_right']
    ], dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return success, rvec, tvec
