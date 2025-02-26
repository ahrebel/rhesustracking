import cv2
from detect_eye import get_eye_coordinates
from section_mapping import create_grid, get_region_for_point
import numpy as np

# Open the video
cap = cv2.VideoCapture("videos/input/1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
delta_t = 1.0 / fps

# Initialize fixation times (one counter per grid cell)
fixation_times = {}

# Read first frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read video.")
    exit(1)
h, w = frame.shape[:2]
grid = create_grid(w, h, n_cols=11, n_rows=10)

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Detect eye coordinate using SLEAP
    x, y = get_eye_coordinates(frame)
    region = get_region_for_point(x, y, grid)
    if region is not None:
        fixation_times[region] = fixation_times.get(region, 0) + delta_t

cap.release()

# Save the fixation times (example: as a CSV)
import csv
with open("data/analysis_output/fixation_times.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Region", "Fixation_Time_sec"])
    for region, t in sorted(fixation_times.items()):
        writer.writerow([region, t])

print("Analysis complete. Fixation times saved.")
