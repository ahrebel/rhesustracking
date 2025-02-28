# head_pose_estimator.py
import numpy as np
import cv2

def estimate_head_pose(landmarks, camera_matrix):
    """
    Estimate head pose using OpenCV's solvePnP.
    For simplicity, we use a minimal set of 3D model points corresponding to key facial landmarks.
    
    landmarks: dict with keys such as "left_pupil" and "right_pupil".
    camera_matrix: Intrinsic camera parameters.
    
    Returns:
        A dictionary containing the rotation and translation vectors.
    """
    # Define 3D model points in an arbitrary coordinate space
    model_points = np.array([
        (0.0, 0.0, 0.0),       # Center point (midpoint between eyes)
        (-30.0, 0.0, -30.0),    # Left eye model point
        (30.0, 0.0, -30.0)      # Right eye model point
    ])

    # Compute image points: use the average for center and the individual pupil points.
    center_x = (landmarks["left_pupil"][0] + landmarks["right_pupil"][0]) / 2
    center_y = (landmarks["left_pupil"][1] + landmarks["right_pupil"][1]) / 2
    image_points = np.array([
        (center_x, center_y),
        landmarks["left_pupil"],
        landmarks["right_pupil"]
    ], dtype="double")

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    
    if not success:
        rotation_vector = np.array([[0.0], [0.0], [0.0]])
        translation_vector = np.array([[0.0], [0.0], [0.0]])
    
    return {
        "rotation_vector": rotation_vector,
        "translation_vector": translation_vector
    }
