# Monkey Gaze Tracker with Touch-Integrated Training

This project provides a Python-based eye-tracking system that analyzes recorded videos of a monkey interacting with a touchscreen. It uses DeepLabCut (DLC) to detect the monkey’s eye landmarks, maps gaze to one of 110 predefined screen sections, and integrates touchscreen (or “click”) event data to refine the gaze mapping model.

The system operates in two modes:
1. **Analysis Mode:** Processes trial videos to compute the total time the monkey’s gaze falls into each screen section. If a corresponding touch event file is present, it correlates touch events with the gaze data.
2. **Training Mode:** Uses paired data (raw eye coordinates and touch locations) to train a regression (or homography) model that updates the mapping from raw eye data to accurate screen coordinates.

## Features
- **Offline Video Processing:** Analyzes pre-recorded trial videos.
- **DeepLabCut Integration:** Uses DLC for eye detection (placeholder provided—you’ll need to integrate your own model/inference).
- **Touch Event Integration:** Loads touch event logs (CSV/TXT) and aligns them with gaze data using timestamps.
- **Gaze Mapping & Calibration:** Divides the screen into 110 sections and computes total fixation time per section. Supports initial calibration and training using paired data.
- **Cross-Platform Compatibility:** Runs on both macOS and Windows.
- **Easy Setup:** Includes setup scripts and a clear folder organization.
- **Training Module:** Update the calibration (gaze mapping) using paired data from video and touch events.



## Setup Instructions

1. **Clone or Download the Repository:**  
   Place all files into a folder (e.g., `monkey-gaze-tracker`).

2. **Set Up Your Python Environment:**
   - **Using Conda (recommended):**
     ```bash
     conda env create -f environment.yml
     conda activate monkey-gaze-tracker
     ```
   - **Using pip and Virtualenv:**
     ```bash
     python -m venv venv
     source venv/bin/activate         # On Windows use: venv\Scripts\activate
     pip install -r requirements.txt
     ```

3. **Run the Provided Setup Script (Optional):**
   - On Mac/Linux:
     ```bash
     bash setup.sh
     ```
   - On Windows:
     ```batch
     setup.bat
     ```

4. **Data Input:**
   - **Videos:** Place your trial video files (e.g., `trial1.mp4`) in the `videos/input/` folder.
   - **Touch Event Files:** For each video, add a corresponding touch event file named with the same base name and the suffix `_touch.csv` (or `.txt`).  
     The touch file should include at least these columns:
     - `timestamp` (time of touch in seconds)
     - `x` (horizontal pixel coordinate of the touch)
     - `y` (vertical pixel coordinate of the touch)

5. **Initial Calibration:**
   Run the calibration script to compute the calibration matrix from predefined calibration points:
   ```bash
   python src/calibrate.py
   ```

6. **Run Video Analysis:**
     Process videos and (if possible) correlate touch events with gaze data:
   ```bash
   python src/analyze_video.py
   ```
   Outputs (CSV and JSON files) will be saved in output/gaze_data/

7. **Training the Gaze Mapping (Optional)**
     To refine your mapping using paired data (raw eye coordinates vs. touch locations), you’ll need a CSV file containing training data with the following columns:
- raw_x, raw_y: The raw eye coordinates (from video frames)
- touch_x, touch_y: The corresponding touch coordinates
   ```bash
   python src/train_gaze_mapping.py --data path/to/your_training_data.csv
   ```
   This will compute a new calibration (gaze mapping) matrix and save it to data/trained_model/calibration_matrix.npy

8. **Visualize Results (Optional)**
       To generate a heatmap of gaze times, run:
     ```bash
      python src/visualize.py
      ```
