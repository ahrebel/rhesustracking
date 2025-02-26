# src/detect_eye.py
import sleap

# Update the path below to point to your exported SLEAP model (ZIP file)
MODEL_PATH = "models/your_trained_model.zip"
sleap_model = sleap.load_model(MODEL_PATH)

def get_eye_coordinates(frame):
    """
    Uses the SLEAP model to detect eye landmark(s) in the given frame.
    Returns the (x, y) coordinate of the detected landmark (e.g., pupil center).
    """
    # SLEAP expects a list of frames
    predictions = sleap_model.predict(frame)
    if predictions and len(predictions) > 0:
        # For single–animal tracking, take the first instance.
        instance = predictions[0]
        # Assume the landmark of interest is the first point (adjust if needed)
        x, y = instance.points[0]
        return (x, y)
    # Fallback: return the center of the frame
    h, w = frame.shape[:2]
    return (w // 2, h // 2)
