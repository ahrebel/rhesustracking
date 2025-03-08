# analyze_gaze.py
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from section_mapping import create_grid, get_region_for_point
from train_gaze_mapping import EyeToScreenNet

def load_pytorch_model(model_path):
    model = EyeToScreenNet(input_dim=8, hidden_dim=16, output_dim=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def map_eye_to_screen(row, model):
    features = np.array([
        row['corner_left_x'], row['corner_left_y'],
        row['corner_right_x'], row['corner_right_y'],
        row['left_pupil_x'],  row['left_pupil_y'],
        row['right_pupil_x'], row['right_pupil_y']
    ], dtype=np.float32)
    input_tensor = torch.from_numpy(features).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze().numpy()  # [screen_x, screen_y]

def analyze_gaze(landmarks_csv, model_path, screen_width, screen_height,
                 n_cols, n_rows, output_heatmap, output_section_csv):
    df = pd.read_csv(landmarks_csv)
    pytorch_model = load_pytorch_model(model_path)

    # Compute screen gaze coords for each frame
    def row_to_screencoords(row):
        return map_eye_to_screen(row, pytorch_model)
    screen_coords = df.apply(row_to_screencoords, axis=1)
    df['screen_x'] = screen_coords.apply(lambda xy: xy[0])
    df['screen_y'] = screen_coords.apply(lambda xy: xy[1])

    # Build the grid
    grid = create_grid(screen_width, screen_height, n_cols, n_rows)

    # Determine which region each frame falls into
    df['region'] = df.apply(lambda row: get_region_for_point(row['screen_x'], row['screen_y'], grid), axis=1)

    # Estimate average frame duration (if 'time' column is present)
    times = df['time'].dropna().values
    if len(times) > 1:
        avg_frame_duration = np.mean(np.diff(times))
    else:
        avg_frame_duration = 0

    # Time spent per region = (#frames in region) * avg_frame_duration
    region_counts = df['region'].value_counts().sort_index()
    region_times = region_counts * avg_frame_duration
    region_times = region_times.reindex(range(len(grid)), fill_value=0).reset_index()
    region_times.columns = ['region', 'time_spent']
    region_times.to_csv(output_section_csv, index=False)
    print(f"Section time data saved to {output_section_csv}")

    # Create a heatmap array
    heatmap_data = np.zeros((n_rows, n_cols))
    for _, row_data in region_times.iterrows():
        region_id = row_data['region']
        if pd.isna(region_id):
            continue
        reg = int(region_id)
        reg_row = reg // n_cols
        reg_col = reg % n_cols
        heatmap_data[reg_row, reg_col] = row_data['time_spent']

    # Plot the heatmap
    plt.figure(figsize=(8,6))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')

    # If you're specifically using a 3x3 grid, label axes more intuitively:
    if n_cols == 3 and n_rows == 3:
        col_labels = ["Left", "Center", "Right"]
        row_labels = ["Top", "Center", "Bottom"]
        plt.xticks(range(n_cols), col_labels)
        plt.yticks(range(n_rows), row_labels)
        # Invert y-axis so row=0 is at the top
        plt.gca().invert_yaxis()

        # Annotate each cell with the time spent
        for i in range(n_rows):
            for j in range(n_cols):
                text_val = f"{heatmap_data[i, j]:.2f}s"
                plt.text(j, i, text_val, ha="center", va="center", color="white")

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
    parser.add_argument("--landmarks_csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--screen_width", type=int, required=True)
    parser.add_argument("--screen_height", type=int, required=True)
    parser.add_argument("--n_cols", type=int, default=3)
    parser.add_argument("--n_rows", type=int, default=3)
    parser.add_argument("--output_heatmap", required=True)
    parser.add_argument("--output_sections", required=True)
    args = parser.parse_args()
    
    analyze_gaze(args.landmarks_csv, args.model, args.screen_width, args.screen_height,
                 args.n_cols, args.n_rows, args.output_heatmap, args.output_sections)
