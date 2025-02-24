import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.config_loader import load_yaml_config

def visualize_heatmap(result_csv, screen_config):
    df = pd.read_csv(result_csv)
    rows = screen_config["rows"]
    cols = screen_config["cols"]
    
    # Create a heatmap array based on section times
    heatmap = np.zeros((rows, cols))
    
    for _, row in df.iterrows():
        section_id = int(row["SectionID"])
        time_spent = row["TimeSpent"]
        # Convert section_id (1-indexed, row-major) to row and col indices
        section_index = section_id - 1
        r = section_index // cols
        c = section_index % cols
        heatmap[r, c] = time_spent
    
    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.title("Gaze Time Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.colorbar(label="Time Spent (s)")
    plt.show()

def main():
    # Example: Visualize the first CSV result in the output folder
    result_dir = "output/gaze_data"
    files = [f for f in os.listdir(result_dir) if f.endswith("_gaze.csv")]
    if not files:
        print("No result CSV files found in output/gaze_data/")
        return
    result_csv = os.path.join(result_dir, files[0])
    screen_config = load_yaml_config("config/screen_config.yaml")
    visualize_heatmap(result_csv, screen_config)

if __name__ == "__main__":
    main()

