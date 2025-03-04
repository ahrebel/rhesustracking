# analyze_gaze.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from section_mapping import create_grid, get_region_for_point

def map_eye_to_screen(row, model):
    """
    Given a DataFrame row with eye landmarks, compute screen coordinates.
    The model expects an 8-element vector:
      [corner_left_x, corner_left_y, corner_right_x, corner_right_y,
       left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y]
    """
    features = np.array([
        row['corner_left_x'], row['corner_left_y'],
        row['corner_right_x'], row['corner_right_y'],
        row['left_pupil_x'], row['left_pupil_y'],
        row['right_pupil_x'], row['right_pupil_y']
    ])
    screen_coords = model.predict(features.reshape(1, -1))
    return screen_coords[0]

def analyze_gaze(landmarks_csv, model_path, screen_width, screen_height, n_cols, n_rows, output_heatmap, output_section_csv):
    # Load the landmarks data
    df = pd.read_csv(landmarks_csv)
    
    # Load the trained regression mapping model
    with open(model_path, 'rb') as f:
        mapping_model = pickle.load(f)
    
    # Compute screen gaze coordinates for each frame
    screen_coords = df.apply(lambda row: map_eye_to_screen(row, mapping_model), axis=1)
    df['screen_x'] = screen_coords.apply(lambda x: x[0])
    df['screen_y'] = screen_coords.apply(lambda x: x[1])
    
    # Create grid sections for the given screen dimensions
    grid = create_grid(screen_width, screen_height, n_cols, n_rows)
    
    # For each frame, determine which section the gaze falls into
    df['region'] = df.apply(lambda row: get_region_for_point(row['screen_x'], row['screen_y'], grid), axis=1)
    
    # Estimate average frame duration from the timestamps
    times = df['time'].dropna().values
    if len(times) > 1:
        avg_frame_duration = np.mean(np.diff(times))
    else:
        avg_frame_duration = 0
    
    # Aggregate total time per region (each appearance counts as one frame duration)
    region_times = df['region'].value_counts().sort_index() * avg_frame_duration
    # Ensure every section is represented (even if zero)
    region_times = region_times.reindex(range(len(grid)), fill_value=0).reset_index()
    region_times.columns = ['region', 'time_spent']
    
    # Save the section time data to a CSV file
    region_times.to_csv(output_section_csv, index=False)
    print(f"Section time data saved to {output_section_csv}")
    
    # Build a heatmap array (grid shape: n_rows x n_cols)
    heatmap_data = np.zeros((n_rows, n_cols))
    for _, row in region_times.iterrows():
        region_id = row['region']
        if pd.isna(region_id):
            continue
        reg = int(region_id)
        reg_row = reg // n_cols
        reg_col = reg % n_cols
        heatmap_data[reg_row, reg_col] = row['time_spent']
    
    # Generate and save the heatmap
    plt.figure(figsize=(8,6))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Time Spent (s)')
    plt.title("Heatmap of Gaze Duration per Screen Section")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(output_heatmap)
    plt.show()
    print(f"Heatmap image saved to {output_heatmap}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map raw eye landmarks to screen gaze coordinates, aggregate gaze duration per section, and output a heatmap."
    )
    parser.add_argument("--landmarks_csv", required=True, help="CSV file with extracted eye landmarks")
    parser.add_argument("--model", required=True, help="Path to trained gaze mapping model (pickle file)")
    parser.add_argument("--screen_width", type=int, required=True, help="Screen width in pixels")
    parser.add_argument("--screen_height", type=int, required=True, help="Screen height in pixels")
    parser.add_argument("--n_cols", type=int, default=3, help="Number of grid columns")
    parser.add_argument("--n_rows", type=int, default=3, help="Number of grid rows")
    parser.add_argument("--output_heatmap", required=True, help="Output path for heatmap image")
    parser.add_argument("--output_sections", required=True, help="Output CSV file for section durations")
    args = parser.parse_args()
    
    analyze_gaze(args.landmarks_csv, args.model, args.screen_width, args.screen_height,
                 args.n_cols, args.n_rows, args.output_heatmap, args.output_sections)

