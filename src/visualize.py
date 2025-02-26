import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def visualize_gaze(csv_file):
    """
    Load gaze points from a CSV file and generate a scatter plot.
    """
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], s=1, alpha=0.5)
    plt.title("Gaze Points")
    plt.xlabel("Screen X")
    plt.ylabel("Screen Y")
    plt.gca().invert_yaxis()  # Invert y-axis if needed to match screen orientation.
    
    output_path = os.path.join('data/analysis_output', 'gaze_visualization.png')
    plt.savefig(output_path)
    plt.show()
    print("Gaze visualization saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Gaze Data')
    parser.add_argument('--csv', type=str, required=True, help='Path to gaze CSV file')
    args = parser.parse_args()
    
    visualize_gaze(args.csv)
