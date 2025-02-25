# Monkey Gaze Tracker with Touch-Integrated Training


** This is an in-progress repo for a rhesus macaque eye tracking software. Please note that it is not finished and will not work as expected. **


Monkey Gaze Tracker is a Python-based eye-tracking system that processes videos of a monkey interacting with a touchscreen. The system leverages DeepLabCut (DLC) to detect eye landmarks, maps gaze positions to one of 110 predefined screen sections, and (optionally) correlates touchscreen events with gaze data to refine the mapping model.

The system supports two primary modes:

1. **Analysis Mode:**  
   Processes trial videos to calculate the total fixation time in each screen section. When a matching touch event file is available, it aligns touch data with the gaze data.

2. **Training Mode:**  
   Uses paired data (raw eye coordinates and touch locations) to train a regression or homography model that improves the accuracy of the gaze-to-screen coordinate mapping.

## Key Features

- **Offline Video Processing:** Analyze pre-recorded trial videos.
- **DeepLabCut Integration:** Uses DLC for eye landmark detection (a placeholder is provided; you can integrate your custom model/inference).
- **Touch Event Correlation:** Aligns touch event logs (CSV/TXT) with gaze data using timestamps.
- **Gaze Mapping & Calibration:** Divides the screen into 110 sections and computes fixation times per section. Supports initial calibration and training using paired data.
- **Cross-Platform Compatibility:** Designed to run on both macOS and Windows.
- **Easy Setup:** Includes setup scripts and an organized folder structure.
- **Training Module:** Update the gaze mapping calibration using paired eye and touch data.


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ahrebel/rhesusDLC
cd rhesusDLC
```


### 2. Set Up Your Python Environment

#### **Using Conda (Recommended):**

```bash
conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
pip install pyyaml
```

#### **Or Using pip with Virtualenv:**

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```


### 3. Prepare Your Data

- **Video Files:**  
  Place your trial video files (e.g., `trial1.mp4`) in the `videos/input/` directory.

- **Touch Event Files:**  
  For each video, provide a corresponding touch event file named with the same base name and a `_touch.csv` (or `.txt`) suffix. This file should include columns such as:
  - `timestamp` (in seconds)
  - `x` (horizontal coordinate)
  - `y` (vertical coordinate)


### 4. Initial Calibration

Generate the calibration matrix using predefined calibration points:

```bash
python src/calibrate.py
```


### 5. Analyze Videos

Process the videos and, if available, correlate touch events with gaze data:

```bash
python src/analyze_video.py
```

Processed output files (CSV/JSON) will be stored in the `output/gaze_data/` folder.


### 6. Train the Gaze Mapping (Optional)

To improve the mapping model using paired data, prepare a CSV file with these columns:

- `raw_x`, `raw_y`: Raw eye coordinates from video frames.
- `touch_x`, `touch_y`: Corresponding touch coordinates.

Then run:

```bash
python src/train_gaze_mapping.py --data path/to/your_training_data.csv
```

This script computes an updated calibration (gaze mapping) matrix and saves it to `data/trained_model/calibration_matrix.npy`.


### 7. Visualize the Results (Optional)

Generate a heatmap to visualize fixation times across the screen:

```bash
python src/visualize.py
```


### 8. Final Verification

To confirm that everything is installed correctly, run:

```bash
python -c "import tables, torch, deeplabcut; print('Installation successful')"
```

Expected output:

```
Loading DLC 2.3.11...
DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)
Installation successful
```
