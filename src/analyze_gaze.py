#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from section_mapping import create_grid, get_region_for_point
from train_gaze_mapping import EyeToScreenNet

def load_pytorch_model(model_path):
    """
    Loads a PyTorch model from file and returns it in evaluation mode.
    """
    model = EyeToScreenNet(input_dim=8, hidden_dim=16, output_dim=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def map_eye_to_screen(row, model):
    """
    Map eye landmarks to screen coordinates.
    
    Expects the row to have columns:
      corner_left_x, corner_left_y, corner_right_x, corner_right_y,
      left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y
    """
    features = np.array([
        row['corner_left_x'], row['corner_left_y'],
        row['corner_right_x'], row['corner_right_y'],
        row['left_pupil_x'],  row['left_pupil_y'],
        row['right_pupil_x'], row['right_pupil_y']
    ], dtype=np.float32)
    
    with torch.no_grad():
        input_tensor = torch.from_numpy(features).unsqueeze(0)  # shape: (1, 8)
        output = model(input_tensor)  # shape: (1, 2)
    return output.squeeze().numpy()  # returns [screen_x, screen_y]

def analyze_gaze(landmarks_csv, model_path, screen_width, screen_height,
                 n_cols, n_rows, output_heatmap, output_sections):
    """
    Reads the landmarks CSV, maps eye landmarks to screen coordinates using the
    trained model, divides the screen into a grid, and computes the time spent 
    in each region to produce a heatmap.
    """
    # Load landmarks CSV (should include a 'time' column)
    df = pd.read_csv(landmarks_csv)
    
    # Load the trained model
    model = load_pytorch_model(model_path)
    
    # Map eye landmarks to screen coordinates
    screen_coords = df.apply(lambda row: map_eye_to_screen(row, model), axis=1)
    df['screen_x'] = screen_coords.apply(lambda coord: coord[0])
    df['screen_y'] = screen_coords.apply(lambda coord: coord[1])
    
    # Create grid of screen sections
    grid = create_grid(screen_width, screen_height, n_cols, n_rows)
    df['region'] = df.apply(lambda r: get_region_for_point(r['screen_x'], r['screen_y'], grid), axis=1)
    
    # Estimate average frame duration from the 'time' column if available
    times = df['time'].dropna().values
    avg_frame_duration = np.mean(np.diff(times)) if len(times) > 1 else 0
    
    # Count frames per region and convert to time spent per region
    region_counts = df['region'].value_counts().sort_index()
    region_times = region_counts * avg_frame_duration
    # Ensure every grid cell has an entry
    region_times = region_times.reindex(range(len(grid)), fill_value=0).reset_index()
    region_times.columns = ['region', 'time_spent']
    region_times.to_csv(output_sections, index=False)
    print(f"Section time distribution saved to {output_sections}")
    
    # Build the heatmap data matrix
    heatmap_data = np.zeros((n_rows, n_cols))
    for _, row_data in region_times.iterrows():
        reg_id = row_data['region']
        if pd.isna(reg_id):
            continue
        reg_id = int(reg_id)  # Cast region id to integer
        reg_row = reg_id // n_cols
        reg_col = reg_id % n_cols
        heatmap_data[reg_row, reg_col] = row_data['time_spent']
    
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Time Spent (s)')
    plt.title("Heatmap of Gaze Duration per Screen Section")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(output_heatmap, dpi=150)
    plt.show()
    print(f"Heatmap saved to {output_heatmap}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze gaze data and generate a heatmap of screen fixation durations.")
    parser.add_argument("--landmarks_csv", required=True,
                        help="CSV file with columns: corner_left_x, corner_left_y, corner_right_x, corner_right_y, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, time")
    parser.add_argument("--model", required=True,
                        help="Path to the trained gaze mapping model (e.g., gaze_mapping_model.pth)")
    parser.add_argument("--screen_width", type=int, required=True, help="Screen width in pixels")
    parser.add_argument("--screen_height", type=int, required=True, help="Screen height in pixels")
    parser.add_argument("--n_cols", type=int, default=3, help="Number of columns to divide the screen")
    parser.add_argument("--n_rows", type=int, default=3, help="Number of rows to divide the screen")
    parser.add_argument("--output_heatmap", required=True, help="Path to save the generated heatmap image")
    parser.add_argument("--output_sections", required=True, help="Path to save the CSV with section durations")
    args = parser.parse_args()
    
    analyze_gaze(args.landmarks_csv, args.model, args.screen_width, args.screen_height,
                 args.n_cols, args.n_rows, args.output_heatmap, args.output_sections)
