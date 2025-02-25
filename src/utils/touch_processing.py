import pandas as pd
from datetime import datetime

def parse_iso_timestamp(ts_str):
    """
    Parse an ISO timestamp string, trimming fractional seconds to 6 digits if necessary.
    Example: "2025-02-24T18:41:57.2864969-05:00" becomes "2025-02-24T18:41:57.286496-05:00".
    """
    try:
        if '.' in ts_str:
            main, frac = ts_str.split('.', 1)
            # Look for timezone sign in the fractional part
            tz_index = None
            for sign in ['+', '-']:
                # Skip a leading '-' if it's part of the fraction
                idx = frac.find(sign, 1)
                if idx != -1:
                    tz_index = idx
                    break
            if tz_index is not None:
                frac_part = frac[:tz_index]
                tz_part = frac[tz_index:]
                # Trim or pad to 6 digits (microseconds)
                frac_part = frac_part[:6].ljust(6, '0')
                ts_str = f"{main}.{frac_part}{tz_part}"
            else:
                frac = frac[:6].ljust(6, '0')
                ts_str = f"{main}.{frac}"
        return datetime.fromisoformat(ts_str)
    except Exception as e:
        raise ValueError(f"Error parsing timestamp '{ts_str}': {e}")

def load_touch_events(file_path):
    """
    Loads touch events from a CSV file.
    The CSV must have the columns: timestamp,x,y.
    The timestamp is expected to be in ISO format.
    This function converts each timestamp to seconds relative to the first timestamp.
    """
    df = pd.read_csv(file_path)
    if df.empty:
        return df
    try:
        df['parsed_timestamp'] = df['timestamp'].apply(parse_iso_timestamp)
        base_time = df.loc[0, 'parsed_timestamp']
        df['timestamp'] = df['parsed_timestamp'].apply(lambda dt: (dt - base_time).total_seconds())
        df.drop(columns=['parsed_timestamp'], inplace=True)
    except Exception as e:
        print("Error converting timestamps:", e)
    return df

def correlate_touch_gaze(gaze_time_series, touch_df, screen_config=None):
    """
    For each touch event in touch_df, finds the most recent gaze event (from gaze_time_series)
    that occurred at or before the touch time.
    Returns a list of dictionaries with touch time, touch coordinates, and the gaze section.
    """
    correlation_results = []
    # Ensure the gaze time series is sorted by time.
    gaze_time_series.sort(key=lambda x: x[0])
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
