import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_gaze_data(gaze_csv):
    return pd.read_csv(gaze_csv)

def load_click_data(click_csv):
    return pd.read_csv(click_csv)

def map_gaze_to_click(gaze_df, click_df, max_time_diff=0.1):
    """
    For each click, find the gaze coordinate nearest in time (within max_time_diff seconds).
    Returns a DataFrame with click times, click positions, and the matched gaze positions.
    """
    mapped = []
    for _, click in click_df.iterrows():
        click_time = click['time']
        # Find gaze entries within the tolerance.
        candidates = gaze_df[np.abs(gaze_df['time'] - click_time) <= max_time_diff]
        if candidates.empty:
            continue
        # Choose the candidate with the smallest time difference.
        closest_idx = (np.abs(candidates['time'] - click_time)).idxmin()
        best_match = candidates.loc[closest_idx]
        mapped.append({
            'click_time': click_time,
            'click_x': click['click_x'],
            'click_y': click['click_y'],
            'gaze_time': best_match['time'],
            'gaze_x': best_match['x'],
            'gaze_y': best_match['y'],
            'time_diff': np.abs(best_match['time'] - click_time)
        })
    return pd.DataFrame(mapped)

def plot_gaze_and_click(mapped_df):
    """
    Plot gaze points over time (colored by time) and overlay click positions.
    """
    plt.figure(figsize=(10, 6))
    # Plot gaze points
    scatter = plt.scatter(mapped_df['gaze_x'], mapped_df['gaze_y'], 
                          c=mapped_df['gaze_time'], cmap='viridis', s=20, label='Gaze')
    # Overlay click points in red.
    plt.scatter(mapped_df['click_x'], mapped_df['click_y'], color='red', marker='x', s=50, label='Click')
    plt.xlabel("Screen X Coordinate")
    plt.ylabel("Screen Y Coordinate")
    plt.title("Gaze Points vs. Click Locations")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Gaze Timestamp (s)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Replace with your actual file paths.
    gaze_csv = "/path/to/your/gaze.csv"   # CSV produced by the video analysis (with timestamps)
    click_csv = "/path/to/your/clicks.csv" # CSV containing your click events and their timestamps

    gaze_df = load_gaze_data(gaze_csv)
    click_df = load_click_data(click_csv)
    
    mapped_df = map_gaze_to_click(gaze_df, click_df, max_time_diff=0.1)
    print("Mapped data (first 5 rows):")
    print(mapped_df.head())
    
    plot_gaze_and_click(mapped_df)
