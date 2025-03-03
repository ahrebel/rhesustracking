# calibrate.py
import numpy as np
import cv2
import yaml

def load_touch_events(touch_file):
    """
    Load touch events from a CSV or TXT file.
    Expects a header line: timestamp,x,y
    Returns a list of (x, y) coordinates.
    """
    events = []
    with open(touch_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                events.append((x, y))
    return events

def calibrate(eye_points, touch_points):
    """
    Compute a homography matrix from raw eye points to touch (screen) points.
    
    Args:
        eye_points: List of (x, y) coordinates from eye tracking.
        touch_points: Corresponding list of (x, y) screen touch points.
    
    Returns:
        Homography matrix H.
    """
    eye_pts = np.array(eye_points, dtype=np.float32)
    touch_pts = np.array(touch_points, dtype=np.float32)
    H, status = cv2.findHomography(eye_pts, touch_pts, cv2.RANSAC)
    return H

if __name__ == "__main__":
    # Example usage:
    # Assume the touch events file (e.g. 'videos/input/1.txt') has a header: timestamp,x,y
    touch_file = 'videos/input/1.txt'
    touch_points = load_touch_events(touch_file)[:4]  # Using first 4 points for calibration
    
    # For demonstration, assume these are the corresponding eye points.
    # In practice, these should come from your calibrated eye tracking data.
    eye_points = [(100, 100), (200, 100), (200, 200), (100, 200)]
    
    H = calibrate(eye_points, touch_points)
    print("Calibration Homography Matrix:")
    print(H)
    
    # Save the calibration matrix.
    model_dir = 'data/trained_model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    np.save(f'{model_dir}/calibration_matrix_1.npy', H)
    with open(f'{model_dir}/calibration_1.yaml', 'w') as f:
        yaml.dump({'homography': H.tolist()}, f)
