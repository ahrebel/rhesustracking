# Rhesus Macaque Gaze Tracker (DeepLabCut Edition)

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye-tracking software for Rhesus macaques using DeepLabCut (DLC) for eye landmark detection. In training, you record videos of the monkey interacting with a touchscreen while also logging the locations of its touches (clicks). The DLC model automatically extracts eye coordinates from the video frames. A calibration procedure maps these raw eye coordinates to screen coordinates. By pairing the calibrated eye positions with the known click locations (using timestamps), you can train a mapping that predicts where on the screen the monkey is looking.

> **Key Features:**
> - **Offline Video Processing:** Analyze pre-recorded trial videos.
> - **DeepLabCut Integration:** Use a trained DLC model to detect eye landmarks.
> - **Touch Event Correlation:** Synchronize touch event logs with gaze data using timestamps.
> - **Calibration & Gaze Mapping:** Compute a calibration matrix to map raw eye data to screen coordinates. Optionally, further refine this mapping using paired training data.
> - **Visualization:** Generate plots to visualize gaze points alongside click events.
> - **Cross-Platform:** Runs on macOS and Windows (with CPU-only setups).

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [Data Preparation](#data-preparation)
3. [DeepLabCut Model Training](#deeplabcut-model-training)
4. [Integration & Eye Detection](#integration--eye-detection)
5. [Calibration](#calibration)
6. [Video Analysis](#video-analysis)
7. [Visualization](#visualization)
8. [Optional: Gaze Mapping Refinement](#optional-gaze-mapping-refinement)
9. [Troubleshooting](#troubleshooting)

---

### Installation and Setup

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/ahrebel/rhesustracking.git
    cd rhesustracking
    ```

2. **Create and Activate Your Python Environment:**

   **Using Conda (recommended):**
    ```bash
    conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
    conda activate monkey-gaze-tracker
    ```

   **Or Using pip (with virtualenv):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

### Data Preparation

- **Video Files:** Place your trial videos (e.g., `1.mp4`, `2.mp4`, …) in `videos/input/`.
- **Touch Event Files:** For each video, have a corresponding CSV (or TXT) file with columns:
    ```csv
    timestamp,x,y
    ```
  Ensure timestamps are in ISO 8601 format.  
- **Gaze Data:** After running the DLC analysis and calibration, your gaze data CSV (e.g. `3DLC_mobnet_100_eyetrackingFeb26shuffle1_100000_labeled_gaze.csv`) should include timestamps along with the x, y coordinates.

---

### Integration and Analysis

1. **Eye Detection and Calibration:**
   - Use your trained DLC model (with its configuration path set in `src/detect_eye.py`) to extract eye coordinates from video frames.
   - Run the calibration script:
     ```bash
     python src/calibrate.py
     ```
   This will generate a calibration matrix that maps raw eye positions to screen coordinates.

2. **Video Analysis:**
   - Process your videos by running:
     ```bash
     python src/analyze_video.py
     ```
   The analysis script:
     - Reads the video.
     - Uses the updated `detect_eye` function to extract eye coordinates (with timestamps).
     - Applies the calibration matrix.
     - Saves the resulting (screen) gaze points to a CSV file in `data/analysis_output/`.

3. **Mapping Gaze to Clicks & Visualization:**
   - Using your saved gaze CSV (with timestamps) and the corresponding click log CSV (with timestamps and click locations), you can now load the data and compare them.
   - See the [Visualization](#visualization) section below.

---

### Visualization

The provided visualization code (in `src/visualize.py`) loads the gaze and click CSV files, then creates a scatter plot. Gaze points are colored by time, and click events are highlighted in red. This helps you visually inspect the mapping and see how much time was spent looking at various screen locations.

---

Happy tracking!
