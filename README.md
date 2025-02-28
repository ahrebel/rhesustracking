# Rhesus Macaque Gaze Tracker (DeepLabCut Edition)

![20250225_2228_Animated Eyes in Cyberspace_remix_01jn04sjzbeb2v028a2akx9nex](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repo creates an eye–tracking software for Rhesus macaques. It is designed to be used for monkeys interacting with a touchscreen using **DeepLabCut (DLC)** for eye–landmark detection. DeepLabCut provides an intuitive GUI for labeling, training, and analysis. Once the DLC model is trained, it can be used to extract eye coordinates from each video frame. These coordinates are then mapped to screen coordinates through calibration which allows for detailed analysis of gaze and fixation patterns and durations.

> **Key Features:**
> - **Offline Video Processing:** Process pre–recorded trial videos for analysis.
> - **DeepLabCut Integration:** Leverage a trained DLC model to detect eye landmarks with a user-friendly GUI.
> - **Touch Event Correlation:** Synchronize touch event logs (CSV/TXT) with gaze data to correlate screen interactions.
> - **Gaze Mapping & Calibration:** Compute a calibration (homography) matrix to accurately map raw eye coordinates to screen coordinates.
> - **Optional Gaze Mapping Training:** Improve mapping accuracy by fine–tuning the calibration model with paired training data.
> - **Visualization:** Generate plots, heatmaps, and other summaries to visualize gaze distribution.
> - **Cross–Platform:** Designed to run on macOS and Windows systems with CPU–only setups.
> - **Enhanced with Head–Pose Estimation:** In addition to eye detection, the system now estimates head pose using additional facial landmarks. This extra step makes the gaze tracker robust against variations in head position and ensures that the mapping from eye–coordinates to screen coordinates remains accurate even when the subject’s head moves.

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Data Preparation](#data-preparation)
3. [DeepLabCut Model Training](#deeplabcut-model-training)
4. [Integration & Eye Detection](#integration--eye-detection)
5. [Calibration](#calibration)
6. [Video Analysis](#video-analysis)
7. [Optional: Fine–Tuning Gaze Mapping](#optional-fine-tuning-gaze-mapping)
8. [Visualization](#visualization)
9. [Data Loading](#data-loading)
10. [Final Verification](#final-verification)
11. [Troubleshooting](#troubleshooting)
12. [Future Improvements](#future-improvements)

---

## Installation and Setup

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/ahrebel/rhesustracking.git
cd rhesustracking
```

### 2. Create and Activate Your Python Environment

#### Using Conda (Recommended):

We recommend using Conda to manage dependencies. Ensure you have Miniconda installed first. Visit the link below to install:
https://www.anaconda.com/download/

Run:

```bash
conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
```

#### Alternatively, Using pip with a Virtual Environment:

If you prefer using pip:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3. Install Required Packages

Install the necessary packages including DeepLabCut by running:

```bash
pip install -r requirements.txt
```

> **Note:**  
> DeepLabCut may install additional dependencies during its setup. If you experience issues, try installing DeepLabCut separately:
> ```bash
> pip install deeplabcut
> pip install pyyaml tensorflow tensorpack tf-slim
> pip install 'deeplabcut[gui]'
> ```
> You may need to run the following (or the version appropriate for your OS):
> ```bash
> pip install --upgrade tensorflow_macos==2.12.0
> ```

---

## Data Preparation

### Video Files

- **Input Videos:**  
  Place your trial video files (e.g., `1.mp4`, `2.mp4`, etc.) in a designated folder (for example, `videos/input/`).

### Touch Event Files

- **Touch Logs:**  
  For each video, create a corresponding touch event file (e.g., `1.txt` or `1.csv`) formatted as CSV with a header:
  ```csv
  timestamp,x,y
  ```
  **Sample Content:**
  ```csv
  2025-02-24T18:41:57.2864969-05:00,16,15
  2025-02-24T18:41:58.6795674-05:00,34,25
  ```
  Make sure timestamps are in ISO 8601 format.

---

## DeepLabCut Model Training

Accurate eye landmark detection is essential. Follow these steps to create and train your DLC model:

### 1. Launch the DLC GUI

Start the DeepLabCut GUI by running:

```bash
python -m deeplabcut
```

### 2. Create a New Project

- In the GUI, select **"Create New Project"**.
- Enter your project name, your name, and add the video(s) you wish to analyze.
- Define the body parts to label (e.g., a single keypoint such as `eye` or `pupil_center`). **For enhanced functionality, label additional keypoints** such as `left_pupil`, `right_pupil`, `corner_left`, and `corner_right` to allow for head–pose estimation.

### 3. Label Frames

- Use the DLC GUI to carefully label the required landmarks on a representative set of frames.
- Aim for diversity in your frame selection to improve model robustness.

### 4. Train the Network

- After labeling, choose the **“Train Network”** option.
- *Tip for CPU–only systems:* Edit your project’s `config.yaml` file to change the network architecture (e.g., from `resnet_50` to a lighter model like `mobilenet_v2_1.0`) to reduce training time and computational load.
- Monitor the training progress until completion.

### 5. Evaluate the Model

- Use the **“Evaluate Network”** option in the GUI to assess detection accuracy.
- If necessary, refine your labels and retrain.

---

## Integration & Eye Detection

### Updated Eye Detection Function

The function in `src/detect_eye.py` has been updated to not only detect the eye coordinate (by, for example, averaging the positions of the left and right pupil) but also to extract additional facial landmarks (such as the eye corners). A new module, `head_pose_estimator.py`, uses these landmarks with OpenCV’s `solvePnP` to estimate head–pose. This helps compensate for head movements during analysis.

The updated function `detect_eye_and_head` does the following:
- Saves a frame as a temporary video file.
- Runs DLC analysis on the frame to generate predictions for the landmarks.
- Extracts the coordinates for `left_pupil`, `right_pupil`, `corner_left`, and `corner_right`.
- Computes the eye coordinate as the average of the pupil positions.
- Defines a camera matrix and calls the head–pose estimator to get rotation and translation vectors.
- Returns a dictionary containing the eye coordinate, the full landmarks, and the head pose.

This extra information is then used in the video analysis pipeline.

---

## Calibration

### Running Calibration

To map raw eye coordinates (which are now potentially adjusted with head–pose information) to screen coordinates, you need a calibration step:

1. **Run Calibration Script:**  
   Execute the calibration script by running:
   ```bash
   python src/calibrate.py
   ```
2. **How It Works:**  
   - The script reads the first four touch events from each touch event file.
   - It computes a homography matrix (calibration matrix) that maps raw eye coordinates to the actual screen coordinates.
3. **Output:**  
   Calibration files (e.g., `calibration_matrix_1.npy` and `calibration_1.yaml`) will be saved in the `data/trained_model/` folder.

---

## Video Analysis

### Analyzing Videos

Once the updated eye detection function is integrated and calibration is complete, you can analyze your videos.

1. **Run Analysis Script:**  
   Execute:
   ```bash
   python src/analyze_video.py
   ```
2. **Processing Details:**  
   - The script processes each video along with its corresponding touch event file.
   - For each frame, it calls the new `detect_eye_and_head` function to extract the raw eye coordinates and head–pose parameters.
   - The calibration matrix is applied to translate raw eye coordinates to screen coordinates.
   - The CSV output now includes the following columns: frame number, raw eye coordinates (`eye_x`, `eye_y`), mapped screen coordinates (`screen_x`, `screen_y`), and head–pose parameters (`rvec_x`, `rvec_y`, `rvec_z`, `tvec_x`, `tvec_y`, `tvec_z`).
3. **Output:**  
   Results are saved in the `data/analysis_output/` directory in CSV format. These data can later be correlated with touch event logs for gaze–tracking analysis.

---

## Optional: Fine–Tuning Gaze Mapping

If you have additional training data with paired information (`raw_x, raw_y, touch_x, touch_y`), you can further refine the gaze mapping:

1. **Run Gaze Mapping Training Script:**
   ```bash
   python src/train_gaze_mapping.py --data path/to/your_training_data.csv
   ```
2. **Result:**  
   This updates the calibration (gaze mapping) model and saves the refined calibration matrix to the `data/trained_model/` folder.

---

## Visualization

Generate plots or heatmaps to visualize fixation data and screen–area distributions:

1. **Run the Visualization Script:**
   ```bash
   python src/visualize.py --csv path/to/your_gaze_data.csv
   ```
2. **Output:**  
   Visualizations (such as heatmaps or summary plots) will be generated to help interpret the fixation distribution across the screen.

---

## Data Loading

A new helper file, `data_loader.py`, provides functions to load gaze data and click event data. For example:

```python
# data_loader.py
import pandas as pd

def load_gaze_data(gaze_csv_path):
    """
    Load gaze data CSV file. Expected columns include:
      frame, eye_x, eye_y, screen_x, screen_y, rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z
    Returns a pandas DataFrame.
    """
    return pd.read_csv(gaze_csv_path)

def load_click_data(click_csv_path):
    """
    Load click event data CSV file. Expected to have columns:
      timestamp,x,y
    Returns a pandas DataFrame.
    """
    return pd.read_csv(click_csv_path)
```

These functions make it easy to later correlate gaze data with touch/click logs.

---

## Final Verification

After installation and setup, verify that all components are installed correctly by running:

```bash
python -c "import tables, deeplabcut; print('Installation successful')"
```

You should see the following output:

```
Installation successful
```

---

## Troubleshooting

- **DeepLabCut Model Issues:**  
  Ensure that the DLC model path in your configuration is correct and that the model has been properly trained.
- **Calibration Errors:**  
  Verify that your touch event files are formatted correctly and share the same base name as their corresponding video files.
- **Performance Problems:**  
  For CPU–only systems, consider reducing the video resolution or switching to a lighter network architecture (e.g., `mobilenet_v2`) in your DLC config.
- **Dependency Conflicts:**  
  If you run into package conflicts, double-check the package versions in `requirements.txt` and install DeepLabCut last to allow it to resolve its own dependencies.

---

## Future Improvements

- **Head–Pose Robustness:**  
  Although head–pose estimation is integrated, further refinement of the 3D facial model and calibration (including per–session camera calibration) could improve accuracy when the subject moves their head.
- **Learning-Based Gaze Mapping:**  
  Integrate a learning module that uses paired training data (raw eye positions and corresponding touch events) to refine the mapping model and potentially adapt to new conditions automatically.
- **Extended Visualization:**  
  Expand visualization tools to include dynamic heatmaps, fixation duration histograms, and region–based analysis to quantify screen interaction patterns.

---

Happy tracking!