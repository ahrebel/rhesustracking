# analyze_gaze.py
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from section_mapping import create_grid, get_region_for_point

# 1) Import or define your EyeToScreenNet architecture
#    If your 'train_gaze_mapping.py' is in src/, you can do:
from train_gaze_mapping import EyeToScreenNet

def load_pytorch_model(model_path):
    """
    Loads a PyTorch model's state_dict from the specified file,
    creates an EyeToScreenNet with the same architecture, and
    populates it with the saved weights.
    """
    # Make sure these dimensions match how you trained your model.
    model = EyeToScreenNet(input_dim=8, hidden_dim=16, output_dim=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def map_eye_to_screen(row, model):
    """
    Given a DataFrame row with eye landmarks, compute screen coordinates.
    We pass the 8-element vector through the PyTorch model.

    The 8 features are:
      [corner_left_x, corner_left_y, corner_right_x, corner_right_y,
       left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y]
    """
    features = np.array([
        row['corner_left_x'], row['corner_left_y'],
        row['corner_right_x'], row['corner_right_y'],
        row['left_pupil_x'],  row['left_pupil_y'],
        row['right_pupil_x'], row['right_pupil_y']
    ], dtype=np.float32)
    
    # Model expects a batch dimension, so unsqueeze(0).
    input_tensor = torch.from_numpy(features).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)  # shape: (1, 2)
    # Convert from tensor to numpy array: shape (2,)
    screen_coords = output.squeeze().numpy()
    return screen_coords  # e.g. [x, y]

def analyze_gaze(landmarks_csv, model_path, screen_width, screen_height,
                 n_cols, n_rows, output_heatmap, output_section_csv):
    # 2) Load the landmarks data
    df = pd.read_csv(landmarks_csv)
    
    # 3) Load the trained PyTorch model
    pytorch_model = load_pytorch_model(model_path)
    
    # 4) Compute screen gaze coordinates for each frame
    def row_to_screencoords(row):
        return map_eye_to_screen(row, pytorch_model)

    screen_coords = df.apply(row_to_screencoords, axis=1)
    # 'screen_coords' is a Series of [x, y] arrays. Let's split them into separate columns.
    df['screen_x'] = screen_coords.apply(lambda xy: xy[0])
    df['screen_y'] = screen_coords.apply(lambda xy: xy[1])
    
    # 5) Create grid sections for the given screen dimensions
    grid = create_grid(screen_width, screen_height, n_cols, n_rows)
    
    # 6) For each frame, determine which section the gaze falls into
    df['region'] = df.apply(lambda row: get_region_for_point(row['screen_x'], row['screen_y'], grid), axis=1)
    
    # 7) Estimate average frame duration from the timestamps (if present)
    times = df['time'].dropna().values
    if len(times) > 1:
        avg_frame_duration = np.mean(np.diff(times))
    else:
        avg_frame_duration = 0
    
    # 8) Aggregate total time per region (each appearance counts as one frame duration)
    region_counts = df['region'].value_counts().sort_index()
    region_times = region_counts * avg_frame_duration
    
    # Ensure every section is represented (even if zero)
    region_times = region_times.reindex(range(len(grid)), fill_value=0).reset_index()
    region_times.columns = ['region', 'time_spent']
    
    # 9) Save the section time data to a CSV file
    region_times.to_csv(output_section_csv, index=False)
    print(f"Section time data saved to {output_section_csv}")
    
    # 10) Build a heatmap array (grid shape: n_rows x n_cols)
    heatmap_data = np.zeros((n_rows, n_cols))
    for _, row in region_times.iterrows():
        region_id = row['region']
        if pd.isna(region_id):
            continue
        reg = int(region_id)
        reg_row = reg // n_cols
        reg_col = reg % n_cols
        heatmap_data[reg_row, reg_col] = row['time_spent']
    
    # 11) Generate and save the heatmap
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
    parser.add_argument("--model", required=True, help="Path to the PyTorch model state_dict (e.g. gaze_mapping_model.pkl)")
    parser.add_argument("--screen_width", type=int, required=True, help="Screen width in pixels")
    parser.add_argument("--screen_height", type=int, required=True, help="Screen height in pixels")
    parser.add_argument("--n_cols", type=int, default=3, help="Number of grid columns")
    parser.add_argument("--n_rows", type=int, default=3, help="Number of grid rows")
    parser.add_argument("--output_heatmap", required=True, help="Output path for heatmap image")
    parser.add_argument("--output_sections", required=True, help="Output CSV file for section durations")
    args = parser.parse_args()
    
    analyze_gaze(args.landmarks_csv, args.model, args.screen_width, args.screen_height,
                 args.n_cols, args.n_rows, args.output_heatmap, args.output_sections)
