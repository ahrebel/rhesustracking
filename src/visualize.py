#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def visualize_gaze(csv_file):
    df = pd.read_csv(csv_file)
    x = df["calibrated_x"].values
    y = df["calibrated_y"].values
    plt.figure(figsize=(8,6))
    plt.hexbin(x, y, gridsize=50, cmap="inferno", mincnt=1)
    plt.colorbar(label="Fixation Count")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Gaze Heatmap")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize gaze data from CSV.")
    parser.add_argument("--csv", required=True, help="Path to gaze CSV file.")
    args = parser.parse_args()
    visualize_gaze(args.csv)

if __name__ == "__main__":
    main()
