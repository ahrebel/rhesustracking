#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import os

def combine_gaze_click(gaze_csv, click_file, output_csv, max_time_diff=0.05):
    # merges gaze data (landmarks) with your click data by time
    # ...
    pass  # (Same code as before; no changes needed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze_csv", required=True)
    parser.add_argument("--click_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_time_diff", type=float, default=0.05)
    args = parser.parse_args()

    combine_gaze_click(args.gaze_csv, args.click_file, args.output_csv, args.max_time_diff)
