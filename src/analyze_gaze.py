#!/usr/bin/env python
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
    with torch.no_grad():
        output = model(torch.from_numpy(features).unsqueeze(0))
    return output.squeeze().numpy()  # [screen_x, screen_y]

def analyze_gaze(landmarks_csv, model_path, screen_width, screen_height,
                 n_cols, n_rows, output_heatmap, output_section_csv):
    df = pd.read_csv(landmarks_csv)
    pytorch_model = load_pytorch_model(model_path)

    screen_coords = df.apply(lambda row: map_eye_to_screen(row, pytorch_model), axis=1)
    df['screen_x'] = screen_coords.apply(lambda xy: xy[0])
    df['screen_y'] = screen_coords.apply(lambda xy: xy[1])

    grid = create_grid(screen_width, screen_height, n_cols, n_rows)
    df['region'] = df.apply(lambda r: get_region_for_point(r['screen_x'], r['screen_y'], grid), axis=1)

    times = df['time'].dropna().values
    avg_frame_duration = np.mean(np.diff(times)) if len(times) > 1 else 0

    region_counts = df['region'].value_counts().sort_index()
    region_times = region_counts * avg_frame_duration
    region_times = region_times.reindex(range(len(grid)), fill_value=0).reset_index()
    region_times.columns = ['region','time_spent']
    region_times.to_csv(output_section_csv, index=False)

    heatmap_data = np.zeros((n_rows, n_cols))
    for _, row_data in region_times.iterrows():
        reg_id = row_data['region']
        if pd.isna(reg_id):
            continue
        reg_row = reg_id // n_cols
        reg_col = reg_id % n_cols
        heatmap_data[reg_row, reg_col] = row_data['time_spent']

    plt.figure(figsize=(8,6))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Time Spent (s)')
    plt.title("Heatmap of Gaze Duration per Screen Section")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(output_heatmap)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
