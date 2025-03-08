# Rhesus Macaque Gaze Tracker (DeepLabCut)

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye–tracking system for Rhesus macaques (and humans) using **DeepLabCut (DLC)**. The system detects eye landmarks from video frames, maps raw eye coordinates to screen positions (via a trained regression model), and analyzes gaze/fixation patterns during touchscreen interactions.

> **Key Features:**
>
> - **Offline Video Processing:** Analyze pre–recorded trial videos.
> - **DeepLabCut–Based Landmark Detection:** Use a trained DLC model to detect key eye landmarks (pupils and corners).
> - **Gaze Mapping:** Use a trained regression model (PyTorch) to accurately translate eye coordinates to screen coordinates.
> - **Visualization:** Generate heatmaps and CSV summaries of time spent looking at each screen region.
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
7. [Step 3: Train the Gaze Mapping Model](#step-3-train-the-gaze-mapping-model)  
8. [Step 4: Process Experimental Videos (Extract Landmarks)](#step-4-process-experimental-videos-extract-landmarks)  
9. [Step 5: Analyze Gaze (Generate Heatmaps & Time Spent)](#step-5-analyze-gaze-generate-heatmaps--time-spent)  
10. [Fine-Tuning or Retraining the Regression Model](#fine-tuning-or-retraining-the-regression-model)  
11. [Troubleshooting](#troubleshooting)  
12. [Future Improvements](#future-improvements)

---

## 1. Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/ahrebel/rhesustracking.git
cd rhesustracking
```

### Create and Activate Your Python Environment

#### Using Conda (Recommended):

```bash
conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
```

#### Or, Using pip with a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

#### Additional Dependencies for DeepLabCut:

```bash
pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
```

> **Note:**  
> If you encounter a `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'` error, run:
> ```bash
> pip install --upgrade tensorflow_macos==2.12.0
> ```
> (Adjust accordingly for non‐macOS systems or different TensorFlow versions.)

---

## 2. Data Preparation

1. **Video Files**  
   Place your trial videos (e.g., `1.mp4`, `2.mp4`, etc.) in a directory of your choice, for example `videos/input/`.

2. **(Optional) Touch/Click Event Files**  
   If you are using additional calibration data from touches or clicks, store them similarly (e.g., `1.csv`, `2.csv`) with columns like `time, screen_x, screen_y` (or `time, click_x, click_y`). This is only necessary if you want to link each frame’s gaze to known on‐screen locations.

---

## 3. DeepLabCut Model Training

1. **Launch the DLC GUI:**

   ```bash
   python -m deeplabcut
   ```

2. **Create a New Project:**  
   - Enter your project name and add your video(s).  
   - Label keypoints such as `left_pupil`, `right_pupil`, `corner_left`, and `corner_right`.

3. **Label Frames:**  
   - Use a diverse set of frames (varying head positions, lighting, etc.).

4. **Train the Network:**  
   - Use the **“Train Network”** option. For CPU-only systems, consider selecting a lighter model (e.g., `mobilenet_v2_1.0`).

5. **Evaluate the Model:**  
   - Run **“Evaluate Network”** to check detection accuracy.

6. **Update Config Path:**  
   - In your scripts (e.g., `detect_eye.py`), set the path to your DLC project’s `config.yaml` so that it can locate your trained model.

---

## 4. Pipeline Overview

1. **(Calibration)** Process a short calibration video to extract eye landmarks at known screen positions.  
2. **(Optional)** Merge the raw gaze data with click data (if your known screen locations come from click/touch events).  
3. **Train the regression model** that maps eye landmarks → screen coordinates.  
4. **Process experimental videos** to extract eye landmarks.  
5. **Analyze gaze** using your trained regression model, producing heatmaps and a CSV of time spent in each screen region.

---

## 5. Step 1: Extract Eye Landmarks for Calibration

You need a **calibration video** for which you know the actual screen coordinates you want the subject to look at (e.g., 4–9 target points around the screen).

1. **Record or gather a calibration video** showing the subject fixating on known screen positions (or times).
2. **Run `process_video.py`** on this calibration video to extract raw landmarks:

   ```bash
   python src/process_video.py \
       --video /path/to/calibration_video.mp4 \
       --config /path/to/dlc_config.yaml \
       --output calibration_landmarks.csv
   ```

   Example:
   ```bash
   python src/process_video.py \
       --video /Users/anthonyrebello/rhesustracking/videos/input/3.mp4 \
       --config /Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml \
       --output landmarks_output.csv
   ```

   This produces a CSV with columns like:  
   ```
   frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
   corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle
   ```

---

## 6. Step 2: (Optional) Merge Gaze Data with Click Data

If your known screen coordinates come from **click events** at specific times, you can merge them with the raw gaze data by matching timestamps.

**Usage Example**:
```bash
python src/combine_gaze_click.py \
  --gaze_csv landmarks_output.csv \
  --click_file /Users/anthonyrebello/rhesustracking/videos/input/3.txt \
  --output_csv calibration_data_for_training.csv \
  --max_time_diff 0.1
```

After merging, your final calibration CSV should have columns:
```
left_corner_x, left_corner_y,
right_corner_x, right_corner_y,
left_pupil_x, left_pupil_y,
right_pupil_x, right_pupil_y,
screen_x, screen_y
```

---

## 7. Step 3: Train the Gaze Mapping Model

Once you have a **calibration CSV** containing both **eye landmarks** and **true screen coordinates**, train the PyTorch regression model:

```bash
python src/train_gaze_mapping.py \
    --data calibration_data_for_training.csv \
    --output data/trained_model/gaze_mapping_model.pth
```

This script produces a `.pth` file containing the trained mapping model (state dictionary).

---

## 8. Step 4: Process Experimental Videos (Extract Landmarks)

Now that you have both a **trained DLC model** (for detecting landmarks) and a **trained PyTorch regression model** (for mapping to screen coords), you can process your real experimental videos:

```bash
python src/process_video.py \
    --video /path/to/experimental_video.mp4 \
    --config /path/to/dlc_config.yaml \
    --output landmarks_output.csv
```

You get a CSV (e.g. `landmarks_output.csv`) with the same columns as in calibration (pupil/corner positions), but no `(screen_x, screen_y)` columns yet.

---

## 9. Step 5: Analyze Gaze (Generate Heatmaps & Time Spent)

Finally, use **`analyze_gaze.py`** to map the raw landmark CSV to actual screen coordinates, split the screen into sections, and create:

1. A **heatmap** image (`.png`).
2. A CSV summarizing **time spent** in each region.

```bash
python src/analyze_gaze.py \
    --landmarks_csv landmarks_output.csv \
    --model data/trained_model/gaze_mapping_model.pth \
    --screen_width 1920 \
    --screen_height 1080 \
    --n_cols 3 \
    --n_rows 3 \
    --output_heatmap gaze_heatmap.png \
    --output_sections section_durations.csv
```

**Parameters**:
- `--landmarks_csv`: The CSV from `process_video.py`.  
- `--model`: The `.pth` file from Step 3.  
- `--screen_width` / `--screen_height`: Dimensions of your screen (or area of interest).  
- `--n_cols` / `--n_rows`: How many columns/rows to divide the screen into.  
- `--output_heatmap`: Where to save the final heatmap image.  
- `--output_sections`: Where to save the CSV listing time spent in each region.

This script:
1. Loads the raw landmarks CSV.
2. Uses the regression model to compute `(screen_x, screen_y)` for each frame.
3. Divides the screen into a grid (via `section_mapping.py`).
4. Sums time spent in each section.
5. Saves both a `.png` heatmap and a `.csv` time summary.

---

## 10. Fine-Tuning or Retraining the Regression Model

You may gather **additional** calibration data over time. Two common ways to incorporate this:

1. **Retrain from Scratch with Combined Data**  
   - Combine old and new calibration data in one CSV (with the same columns).
   - Re–run the training script:
     ```bash
     python src/train_gaze_mapping.py \
         --data combined_calibration_data.csv \
         --output data/trained_model/gaze_mapping_model.pth
     ```
   - This approach is typically simpler because the model sees **all** data at once.

2. **Incremental / Fine‐Tuning**  
   - For a PyTorch model, you can load the saved `state_dict` and continue training with new data. This can be faster but watch for **catastrophic forgetting** if you train only on new data.

Often, method #1 (combined retraining) is easiest unless your dataset is extremely large.

---

## 11. Troubleshooting

- **DLC Detection Problems**:
  - Make sure your DLC project is properly trained. Check labeling consistency and run Evaluate Network.
- **Mapping Accuracy**:
  - Confirm that each row in your calibration CSV is correct: the `(screen_x, screen_y)` truly aligns with the eye landmarks in that row.
  - Collect calibration data spanning a variety of angles and distances for robust results.
- **Dependency Issues**:
  - Verify that your environment matches the `requirements.txt` (especially TensorFlow, PyTorch, and DLC versions).
- **TensorFlow/Keras Errors**:
  - If you see `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'`, upgrade or reinstall TensorFlow as described in [Installation and Setup](#installation-and-setup).

---

## 12. Future Improvements

- **Advanced Head–Pose Estimation**:
  - Incorporate more robust 3D facial landmarks for better yaw/pitch/roll compensation.
- **Adaptive Gaze Mapping**:
  - Develop an online calibration or real–time incremental training approach.
- **Interactive Visualization**:
  - Create dynamic dashboards or GUIs to visualize fixations in real time.

---

**Happy tracking!**
