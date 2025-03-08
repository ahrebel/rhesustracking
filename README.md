# Rhesus Macaque Gaze Tracker (DeepLabCut)

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye–tracking system for Rhesus macaques (and humans) using **DeepLabCut (DLC)**. The system detects eye landmarks from video frames, maps raw eye coordinates to screen positions (via a trained regression model), and analyzes gaze/fixation patterns during touchscreen interactions.

> **Key Features:**
>
> - **Offline Video Processing:** Analyze pre–recorded trial videos.
> - **DeepLabCut–Based Landmark Detection:** Use a trained DLC model to detect key eye landmarks (pupils and corners).
> - **Gaze Mapping:** Use a trained regression model to accurately translate eye coordinates to screen coordinates.
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
11. [Ensuring Correct Screen Dimensions](#ensuring-correct-screen-dimensions)  
12. [Troubleshooting](#troubleshooting)  
13. [Future Improvements](#future-improvements)

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

---

## 2. Data Preparation

1. **Video Files**  
   Place your trial videos (e.g., `1.mp4`, `2.mp4`, etc.) in a directory of your choice (for example, `videos/input/`).

2. **(Optional) Touch/Click Event Files**  
   If you are using additional calibration data from touches or clicks, store them similarly (e.g., `1.csv`, `2.csv`) with columns like `time, screen_x, screen_y` or `time, click_x, click_y`. This step is only necessary if you plan to do a more advanced approach to link each frame’s gaze with a known on‐screen location.

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
   - Use the **"Train Network"** option. For CPU-only systems, consider selecting a lighter model (e.g., `mobilenet_v2_1.0`).

5. **Evaluate the Model:**  
   - Run **"Evaluate Network"** to verify detection accuracy.

6. **Update Config Path:**  
   - In your `detect_eye.py` (or similar script), set the path to your DLC project’s `config.yaml` so that it can locate your trained model.

---

## 4. Pipeline Overview

Below is the recommended workflow:

1. **(Calibration) Process a short calibration video** to extract eye landmarks at known screen positions (or times).  
2. **(Optional)** Merge the raw gaze data with click data (if your known on‐screen locations are from click events).  
3. **Train the regression model** that maps eye landmarks → screen coordinates.  
4. **Process your actual experimental videos** to extract eye landmarks.  
5. **Analyze gaze** using your trained regression model, producing heatmaps and a CSV of time spent in each screen section.

---

## 5. Step 1: Extract Eye Landmarks for Calibration

You need a **calibration video** in which you know the actual screen coordinates you want the subject to look at (for example, a few targets in known locations or times).

1. **Record or gather a calibration video** showing the subject fixating on 4–9 known points on the screen.  
2. **Run `process_video.py`** on this calibration video to extract raw landmarks:

   ```bash
   python src/process_video.py \
       --video /path/to/calibration_video.mp4 \
       --config /path/to/dlc_config.yaml \
       --output calibration_landmarks.csv
   ```

   **Example**:
   ```bash
   python src/process_video.py \
       --video /Users/anthonyrebello/rhesustracking/videos/input/3.mp4 \
       --config /Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml \
       --output landmarks_output.csv
   ```

   - This outputs a CSV with columns like `frame, time, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle, ...`

---

## 6. Step 2: (Optional) Merge Gaze Data with Click Data

If your known screen coordinates come from **click events** (or touches) at specific times, you can merge them with the raw gaze data (landmarks) by matching timestamps.

**Example** usage:
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

Once you have a **calibration CSV** that includes both **eye landmarks** and **true screen coordinates**, you can train the regression model:

```bash
python src/train_gaze_mapping.py \
    --data calibration_data_for_training.csv \
    --output data/trained_model/gaze_mapping_model.pkl
```

This script produces a `.pkl` file containing the trained mapping model.

---

## 8. Step 4: Process Experimental Videos (Extract Landmarks)

Now that you have a **trained DLC model** (for detecting landmarks) **and** a **trained regression model** (for mapping landmarks to screen coords), you can process your actual experimental videos:

```bash
python src/process_video.py \
    --video /path/to/experimental_video.mp4 \
    --config /path/to/dlc_config.yaml \
    --output landmarks_output.csv
```

This step outputs a CSV (e.g. `landmarks_output.csv`) containing the same columns as your calibration CSV (pupil/corner positions), but *without* the final `(screen_x, screen_y)` columns.

---

## 9. Step 5: Analyze Gaze (Generate Heatmaps & Time Spent)

Finally, use **`analyze_gaze.py`** to map the raw landmark CSV into actual screen coordinates, split the screen into sections, and produce:

1. A **heatmap** image.
2. A CSV summarizing **time spent** in each region.

```bash
python src/analyze_gaze.py \
    --landmarks_csv landmarks_output.csv \
    --model data/trained_model/gaze_mapping_model.pkl \
    --screen_width 1920 \
    --screen_height 1080 \
    --n_cols 3 \
    --n_rows 3 \
    --output_heatmap gaze_heatmap.png \
    --output_sections section_durations.csv
```

- **`--landmarks_csv`**: The CSV from `process_video.py`.  
- **`--model`**: The `.pkl` file you trained in Step 3.  
- **`--screen_width`** / **`--screen_height`**: Dimensions of your screen or area of interest.  
- **`--n_cols`** / **`--n_rows`**: How many columns/rows to split the screen into.  
- **`--output_heatmap`**: Path to save the generated heatmap image.  
- **`--output_sections`**: Path to save a CSV listing the time spent in each region.

This script:
1. Loads the raw landmarks CSV.
2. Uses the regression model to compute `(screen_x, screen_y)` for each frame.
3. Divides the screen into the specified grid (via `section_mapping.py`).
4. Aggregates total time spent in each section.
5. Saves both a heatmap (`gaze_heatmap.png`) and a CSV (`section_durations.csv`).

---

## 10. Fine-Tuning or Retraining the Regression Model

After training your initial regression model, you might gather **additional** calibration data over time. There are two common ways to incorporate new data:

1. **Retrain from Scratch with Combined Data**  
   - **Combine** your original calibration data and your new data into a single, larger CSV (with the same columns).  
   - **Run** the training script again, pointing it to the combined CSV:
     ```bash
     python src/train_gaze_mapping.py \
         --data combined_calibration_data.csv \
         --output data/trained_model/gaze_mapping_model.pkl
     ```
   - This method typically yields robust results because the model sees **all** data at once, reinforcing what it already learned while integrating the new examples.

2. **Incremental Training / Fine‐Tuning**  
   - If your model (e.g., a PyTorch or scikit‐learn estimator) supports **incremental learning** (like a `partial_fit` method), you can load the existing model and continue training only on the new data.  
   - For a small neural network in PyTorch, you can load the saved `state_dict` and run more training epochs using only the new data or a mix of old + new data.  
   - This approach can be faster if your dataset is very large or if you only have a small batch of new data. However, you have to be careful about **catastrophic forgetting** (where the model “forgets” earlier data if it only sees new data).

For many use cases, **method #1** (retraining from scratch with combined data) is simpler and safer. But if your dataset grows large, or you want real‐time updates, partial or online training can be used.

---

## 11. Ensuring Correct Screen Dimensions

If your click data shows coordinates **larger** or **smaller** than your final screen size, your heatmaps may all end up in one region or out of bounds. Here’s how to ensure your coordinate system is consistent:

1. **Record Click Coordinates in Actual Screen Size**  
   - If you’re on Windows, set `Form1.WindowState = FormWindowState.Maximized` and `FormBorderStyle = None`. Then read `Screen.PrimaryScreen.Bounds.Width` / `.Height` to confirm the resolution.  
   - On macOS, be aware that **Retina** or “scaled” displays can report double‐size coordinates (e.g., 3000×2000 for a “1512×982” logical display).  
   - If your clicks are 2× bigger, **divide** by 2 before saving to match the “logical” resolution, or just **use** the larger (physical) resolution in `analyze_gaze.py`.

2. **Use the Same Width/Height in `analyze_gaze.py`**  
   - If your click data has `screen_x` up to 3024, do `--screen_width 3024 --screen_height 1964`.  
   - If you want a simpler “logical” 1512×982 space, **divide** your click data by 2, then specify 1512×982.

3. **Check the Predicted Coordinates**  
   - If the model outputs x=4000 but your screen width is 1920, everything is out of range. Either scale the data or use a bigger screen dimension.

4. **Example**: VB .NET Full‐Screen  
   ```vbnet
   ' In Form1.Designer.vb:
   Me.FormBorderStyle = FormBorderStyle.None
   Me.WindowState = FormWindowState.Maximized
   Dim sw As Integer = Screen.PrimaryScreen.Bounds.Width
   Dim sh As Integer = Screen.PrimaryScreen.Bounds.Height
   Me.ClientSize = New Size(sw, sh)
   ```
   Then, when you capture clicks, they range from (0,0) to (sw, sh). Use those same values in your Python scripts.

**Bottom line**: Your recorded `(screen_x, screen_y)` must match the **`--screen_width` and `--screen_height`** you use later. If they mismatch, you’ll see out‐of‐range or compressed heatmaps.

---

## 12. Troubleshooting

- **Model Issues / DLC Not Detecting Landmarks Properly:**
  - Verify that your `config.yaml` is correct and your DLC project is fully trained.
- **Mapping Accuracy Problems:**
  - Check that the data used to train your regression model covers a sufficient range of head poses and gaze positions.
  - Make sure your calibration data CSV is aligned: each row’s `(screen_x, screen_y)` truly corresponds to that row’s eye landmarks.
  - **Confirm your screen dimension** is correct in both your click data and your final analysis script.
- **Dependency Issues:**
  - Ensure all package versions match those listed in `requirements.txt`.
- **TensorFlow or Keras Errors:**
  - If you see `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'`, upgrade TensorFlow as shown in [Installation and Setup](#installation-and-setup).

---

## 13. Future Improvements

- **Enhanced Head–Pose Estimation:**
  - Refine the head pose calculations or incorporate more 3D facial landmarks to improve accuracy.
- **Adaptive Gaze Mapping:**
  - Implement an online or continuously updated calibration procedure.
- **Advanced Visualization Tools:**
  - Develop dynamic or interactive dashboards to visualize fixations and heatmaps in real time.

---

**Happy tracking!**  

Feel free to open an issue or submit a pull request if you have suggestions or encounter any problems.
