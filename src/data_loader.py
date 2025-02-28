# data_loader.py
import pandas as pd

def load_gaze_data(gaze_csv_path):
    """
    Load gaze data CSV file. Expected columns include:
      frame, eye_x, eye_y, screen_x, screen_y, rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z
    Returns a pandas DataFrame.
    """
    return pd.read_csv(gaze_csv_path)

def load_click_data(click_csv_path):
    """
    Load click event data CSV file. Expected to have columns:
      timestamp, x, y
    Returns a pandas DataFrame.
    """
    return pd.read_csv(click_csv_path)
