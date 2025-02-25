import pandas as pd

def load_touch_events(file_path):
    """
    Load touch/click data from a CSV or TXT file.
    Assumes the file has a header with the columns: timestamp, x, y.
    Timestamps are parsed as datetime objects.
    """
    try:
        df = pd.read_csv(file_path)
        # Convert the 'timestamp' column to datetime objects.
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception as e:
        print("Error loading touch events from", file_path, ":", e)
        return pd.DataFrame()

def correlate_touch_gaze(gaze_time_series, touch_df, screen_config):
    """
    For each touch event, find the closest gaze measurement (by time)
    and return a dictionary mapping the touch event's ISO timestamp to a dict
    containing:
      - 'touch_time': time in seconds relative to video start,
      - 'closest_gaze_time': the gaze timestamp (in seconds),
