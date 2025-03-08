# Rhesus Macaque Gaze Tracker (DeepLabCut)

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye–tracking system for Rhesus macaques (and humans) using **DeepLabCut (DLC)**. The system detects eye landmarks from video frames, converts raw eye coordinates to screen coordinates using a trained k–Nearest Neighbors (kNN) regression model, and analyzes gaze/fixation patterns during touchscreen interactions.

> **Key Features:**
>
> - **Offline Video Processing:** Analyze pre–recorded trial videos.
> - **DeepLabCut–Based Landmark Detection:** Use a trained DLC model to detect key eye landmarks (pupils and corners).
> - **Gaze Mapping (kNN):** Map eye coordinates to screen coordinates using a kNN regressor trained on calibration data.
> - **Visualization:** Generate heatmaps and CSV summaries of time spent in each screen region.
> - **Cross–Platform Support:** Runs on both macOS and Windows (CPU–only supported).
> - **Optional Head–Pose Estimation:** Includes head roll estimation using facial landmarks.

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)  
2. [Data Preparation](#data-preparation)  
3. [DeepLabCut Model Training](#deeplabcut-model-training)  
4. [Pipeline Overview](#pipeline-overview)  
5. [Step 1: Extract Eye Landmarks for Calibration](#step-1-extract-eye-landmarks-for-calibration)  
6. [Step 2: (Optional) Merge Gaze Data with Click Data](#step-2-optional-merge-gaze-data-with-click-data)  
7. [Step 3: Train the kNN Mapping Model](#step-3-train-the-knn-mapping-model)  
8. [Step 4: Process Experimental Videos (Extract Landmarks)](#step-4-process-experimental-videos-extract-landmarks)  
9. [Step 5: Analyze Gaze (Generate Heatmaps & Time Spent)](#step-5-analyze-gaze-generate-heatmaps--time-spent)  
10. [Fine-Tuning or Retraining the kNN Model](#step-10-fine-tuning-or-retraining-the-knn-model)  
11. [Troubleshooting](#troubleshooting)  
12. [Future Improvements](#future-improvements)

---

## 1. Installation and Setup

- **Clone the repository:**
  ```bash
  git clone https://github.com/ahrebel/rhesustracking.git
  cd rhesustracking
  ```
- **Set up your environment (recommended with Conda):**
  ```bash
  conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
  conda activate monkey-gaze-tracker
  ```
- **Install required packages:**
  ```bash
  pip install -r requirements.txt
  ```
- **Install additional DeepLabCut dependencies:**
  ```bash
  pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
  ```
  > If you encounter a `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'`, run:
  > ```bash
  > pip install --upgrade tensorflow_macos==2.12.0
  > ```
  > Adjust this for non-macOS systems or different TensorFlow versions.

---

## 2. Data Preparation

- **Video Files:**  
  Store your trial videos (e.g., `1.mp4`, `2.mp4`, etc.) in a designated folder such as `videos/input/`.
- **(Optional) Touch/Click Event Files:**  
  If you have additional calibration data (for example, click or touch events), store these CSV files with headers like `time, screen_x, screen_y` (or similar).

---

## 3. DeepLabCut Model Training

1. **Launch the DLC GUI:**
   ```bash
   python -m deeplabcut
   ```
2. **Create a New Project:**  
   Enter your project name, add your video(s), and label keypoints such as `left_pupil`, `right_pupil`, `corner_left`, and `corner_right`.
3. **Label Frames:**  
   Ensure you label frames from various conditions (different head poses, lighting, etc.).
4. **Train and Evaluate:**  
   Train the network (consider lighter models like `mobilenet_v2_1.0` for CPU-only systems) and evaluate its performance.
5. **Configuration Update:**  
   Update the paths to your DLC configuration (e.g., `config.yaml`) in your scripts.

---

## 4. Pipeline Overview

1. **Calibration:** Use a calibration video to extract eye landmarks at known screen positions.
2. **(Optional) Merge Data:** Merge raw gaze data with click/touch data (if available) to form a calibration CSV.
3. **Mapping Model Training:** Train a kNN regression model using your calibration CSV.
4. **Experimental Processing:** Process experimental videos to extract raw landmarks.
5. **Gaze Analysis:** Use the trained kNN model to predict screen coordinates from landmarks, divide the screen into a grid, and generate heatmaps showing fixation durations.

---

## 5. Step 1: Extract Eye Landmarks for Calibration

Run your video processing script to extract eye landmarks:
```bash
python src/process_video.py --video /path/to/calibration_video.mp4 --config /path/to/dlc_config.yaml --output landmarks_output.csv
```
Example:
```bash
python src/process_video.py --video /Users/anthonyrebello/rhesustracking/videos/input/3.mp4 --config /Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml --output landmarks_output.csv
```
The resulting CSV should include columns such as `frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle`.

---

## 6. Step 2: (Optional) Merge Gaze Data with Click Data

If you have click/touch event data with known screen coordinates, merge it with your gaze data:
```bash
python src/combine_gaze_click.py --gaze_csv landmarks_output.csv --click_file /path/to/your_click_file.csv --output_csv calibration_data_for_training.csv --max_time_diff 0.1
```
After merging, the calibration CSV will include:
```
left_corner_x, left_corner_y,
right_corner_x, right_corner_y,
left_pupil_x, left_pupil_y,
right_pupil_x, right_pupil_y,
screen_x, screen_y
```

---

## 7. Step 3: Train the kNN Mapping Model

With your calibration CSV prepared, train a kNN mapping model that uses the eight eye landmark features to predict the 2D screen coordinates. Open your terminal and run:
```bash
python src/train_knn_mapping.py --data calibration_data_for_training.csv --output data/trained_model/knn_mapping_model.joblib --neighbors 5
```
This command trains a kNN regressor (defaulting to 5 neighbors) and saves the trained model using joblib. You can adjust the `--neighbors` parameter as needed.

---

## 8. Step 4: Process Experimental Videos (Extract Landmarks)

Use the same processing script to extract landmarks from your experimental videos:
```bash
python src/process_video.py --video /path/to/experimental_video.mp4 --config /path/to/dlc_config.yaml --output landmarks_output.csv
```
The resulting CSV will contain the raw eye landmarks (and a time column) for your experimental trials.

---

## 9. Step 5: Analyze Gaze (Generate Heatmaps & Time Spent)

Use an analysis script that loads your trained kNN model to convert landmarks into screen coordinates, then divides the screen into a grid and computes fixation durations per region. Run:
```bash
python src/analyze_gaze_knn.py --landmarks_csv landmarks_output.csv --model data/trained_model/knn_mapping_model.joblib --screen_width 1920 --screen_height 1080 --n_cols 3 --n_rows 3 --output_heatmap gaze_heatmap.png --output_sections section_durations.csv
```
This command will:
- Load your landmarks CSV.
- Predict screen coordinates from eye landmarks using the kNN model.
- Divide the screen (e.g., 1920×1080) into a grid (3×3 in this example).
- Compute the time spent in each grid region (using the frame duration derived from the `time` column).
- Generate and save a heatmap image (`gaze_heatmap.png`) and a CSV file (`section_durations.csv`) summarizing fixation durations per region.

---

## 10. Fine-Tuning or Retraining the kNN Model

When additional calibration data becomes available:
- **Combine** the new data with your existing calibration CSV.
- **Retrain the model:**
  ```bash
  python src/train_knn_mapping.py --data combined_calibration_data.csv --output data/trained_model/knn_mapping_model.joblib --neighbors 5
  ```
- **Tune Hyperparameters:**  
  Experiment with different numbers of neighbors (`--neighbors`) to optimize performance.

---

## 11. Troubleshooting

- **Uniform Heatmap (All Regions Look the Same):**  
  - Verify that the raw landmark data from `process_video.py` varies over time.  
  - Check that your calibration CSV covers a wide range of gaze positions.  
  - Use debug prints (e.g., print sample predicted screen coordinates) to ensure the kNN model is outputting varied results.
- **Model Prediction Issues:**  
  If the predicted screen coordinates are nearly constant, consider:
  - Increasing the diversity of your calibration data.
  - Tuning the `--neighbors` parameter.
  - Normalizing your input features.
- **Data Format:**  
  Ensure that the input CSV files have the required columns and that the `time` values are in a proper numeric or datetime format.

---

## 12. Future Improvements

- **Advanced Feature Engineering:**  
  Experiment with additional derived features (e.g., distances, angles) that may improve the mapping.
- **Model Comparisons:**  
  Compare kNN with other regression techniques (SVR, Random Forests, etc.) using cross-validation.
- **Adaptive Calibration:**  
  Develop an online or real-time calibration approach to continuously refine the mapping model.
- **Enhanced Visualization:**  
  Create interactive dashboards or real-time displays for fixation visualization.

---

**Happy Tracking!**
