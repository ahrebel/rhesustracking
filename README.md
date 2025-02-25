# Rhesus Macaque Gaze Tracker with Touch-Integrated Training

**Note: This repository is a work in progress. The code and instructions provided here may require further tuning and data for optimal performance.**

Monkey Gaze Tracker is a Python-based eye-tracking system that processes videos of a monkey interacting with a touchscreen. The system leverages DeepLabCut (DLC) to detect eye landmarks, maps raw eye positions to one of 110 predefined screen sections, and (optionally) correlates touchscreen events with gaze data to refine the mapping model.

There are two main modes in this project:

1. **Analysis Mode**  
   Processes trial videos to calculate the total fixation time in each screen section. When a matching touch event file is available, it aligns the touch data with the gaze data.

2. **Training Mode**  
   Uses paired data (raw eye coordinates and corresponding touch locations) to train a regression/homography model that improves the accuracy of the gaze-to-screen coordinate mapping.

Additionally, you must train your DeepLabCut (DLC) model to detect the relevant eye landmarks. (The gaze mapping calibration and analysis assume you have a trained DLC model that produces reliable eye coordinates.)

---

## Key Features

- **Offline Video Processing:** Analyze pre-recorded trial videos.
- **DeepLabCut Integration:** Uses DLC for eye landmark detection. (Training DLC is a separate process—see below.)
- **Touch Event Correlation:** Aligns touch event logs (CSV or TXT) with gaze data using timestamps.
- **Gaze Mapping & Calibration:** Divides the screen into 110 sections and computes fixation times per section. Supports initial calibration and training using paired data.
- **Cross-Platform Compatibility:** Designed for both macOS and Windows.
- **Easy Setup:** Includes setup scripts and an organized folder structure.
- **Training Module:** Update the gaze mapping calibration using paired eye and touch data.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ahrebel/rhesusDLC
cd rhesusDLC
```

### 2. Set Up Your Python Environment

#### Using Conda (Recommended):

```bash
conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
pip install pyyaml deeplabcut
```

#### Or Using pip with Virtualenv:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Your Data

- **Video Files:**  
  Place your trial video files (e.g., `1.mp4`, `trial1.mp4`, etc.) in the `videos/input/` directory.

- **Touch Event Files:**  
  For each video, provide a corresponding touch event file named with the same base name. For example, for `1.mp4` use `1.txt` (or `1_touch.csv`). The file should contain a header and rows with the following columns:

  ```
  timestamp,x,y
  2025-02-24T18:41:57.2864969-05:00,16,15
  2025-02-24T18:41:58.6795674-05:00,34,25
  ... etc.
  ```

  (Make sure the timestamps are in ISO 8601 format, even though slight differences in precision may occur.)

### 4. Initial Calibration

Run the calibration script to compute a calibration (homography) matrix that maps raw eye coordinates to screen coordinates. (This uses the first four touch events.)

```bash
python src/calibrate.py
```

For a file named `1.mp4` with touch data in `1.txt`, the calibration files will be saved as:
- `data/trained_model/calibration_matrix_1.npy`
- `data/trained_model/calibration_1.yaml`

### 5. Analyze Videos

Run the analysis script to process your videos. This script will:
- Read the video,
- Extract eye coordinates (using your DLC model’s output; a placeholder currently returns the center of the frame),
- Apply the calibration matrix,
- Map the gaze to screen sections,
- Save fixation time results as CSV and JSON,
- And (if a corresponding touch file exists) correlate touch events with gaze data.

```bash
python src/analyze_video.py
```

*Note:* If your video “1.mp4” has an associated touch data file named `1.txt`, the script will now correctly load and process that file instead of reporting “No touch file found.”

### 6. Train the Gaze Mapping (Optional)

If you have paired data (raw eye coordinates and touch locations) in a CSV file (with columns: `raw_x, raw_y, touch_x, touch_y`), run:

```bash
python src/train_gaze_mapping.py --data path/to/your_training_data.csv
```

This will update the calibration (gaze mapping) matrix and save it to the `data/trained_model/` directory (e.g., as `calibration_matrix.npy`).

### 7. Visualize Results (Optional)

Generate a heatmap of gaze fixations:

```bash
python src/visualize.py
```

### 8. Training the DeepLabCut (DLC) Model

**Important:** Training the DLC model is separate from the gaze mapping calibration. You must train DLC on labeled video frames showing the monkey’s eye landmarks. Follow these steps:

1. **Create a New DLC Project:**

   ```bash
   python -m deeplabcut.create_new_project --project MonkeyGaze --experimenter YourName --videos path/to/your_video.mp4 --working_directory path/to/your_project_folder
   ```

   This creates a project folder (with a configuration file, e.g., `config.yaml`).

2. **Label Frames:**

   Open the labeling GUI to manually annotate the eye landmarks on selected frames:

   ```bash
   python -m deeplabcut.label_frames path/to/config.yaml
   ```

3. **Extract Additional Frames (Optional):**

   If needed, extract more frames for labeling:

   ```bash
   python -m deeplabcut.extract_frames path/to/config.yaml
   ```

4. **Train the Network:**

   Once enough frames are labeled, start training the network:

   ```bash
   python -m deeplabcut.train_network path/to/config.yaml
   ```

   Adjust training parameters in `config.yaml` as needed.

5. **Evaluate the Model:**

   After training, evaluate the network’s performance:

   ```bash
   python -m deeplabcut.evaluate_network path/to/config.yaml
   ```

6. **Analyze New Videos with DLC:**

   When satisfied with performance, use the trained model to analyze new videos:

   ```bash
   python -m deeplabcut.analyze_videos path/to/config.yaml --videos path/to/new_video.mp4
   ```

These steps are described in more detail in the [DeepLabCut documentation](https://deeplabcut.github.io/DeepLabCut/docs/).

### 9. Final Verification

To ensure everything is installed correctly, run:

```bash
python -c "import tables, torch, deeplabcut; print('Installation successful')"
```

You should see output similar to:

```
Loading DLC 2.3.11...
DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)
Installation successful
```
