import pandas as pd
from datetime import datetime

def load_touch_events(file_path):
    """
    Loads touch events from a CSV file.
    The CSV must have the columns: timestamp,x,y
    Timestamps are expected in ISO format. This function converts them
    to seconds relative to the first timestamp.
    """
    df = pd.read_csv(file_path)
    if df.empty:
        return df
    try:
        # Convert timestamp strings to datetime objects and then to relative seconds.
        base_time = datetime.fromisoformat(df.loc[0, "timestamp"])
        df["timestamp"] = df["timestamp"].apply(
            lambda ts: (datetime.fromisoformat(ts) - base_time).total_seconds()
        )
    except Exception as e:
        print("Error converting timestamps:", e)
    return df

def correlate_touch_gaze(gaze_time_series, touch_df, screen_config=None):
    """
    Correlates each touch event with the corresponding gaze event.
    For each touch event (from touch_df), it finds the most recent gaze event (from gaze_time_series)
    that occurred at or before the touch time.
    Returns a list of dictionaries with the touch time, touch coordinates, and the gaze section.
    """
    correlation_results = []
    # Assuming gaze_time_series is sorted by time (each element is (time, section_id))
    for _, row in touch_df.iterrows():
        touch_time = row["timestamp"]
        touch_x = row["x"]
        touch_y = row["y"]
        matching_section = None
        for t, section in gaze_time_series:
            if t <= touch_time:
                matching_section = section
            else:
                break
        correlation_results.append({
            "touch_time": touch_time,
            "touch_x": touch_x,
            "touch_y": touch_y,
            "gaze_section": matching_section
        })
    return correlation_results
