# Rhesus Macaque Gaze Tracker (DeepLabCut)

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye–tracking system for Rhesus macaques (and humans) using **DeepLabCut (DLC)**. The system detects eye landmarks from video frames, maps raw eye coordinates to screen positions (via a trained regression model), and analyzes gaze/fixation patterns during touchscreen interactions.

> **Key Features:**
>
> - **Offline Video Processing:** Analyze pre–recorded trial videos.
> - **DeepLabCut–Based Landmark Detection:** Use a trained DLC model to detect key eye landmarks (pupils and corners).
> - **Gaze Mapping:** Use a trained regression model (optional) to accurately translate eye coordinates to screen coordinates.
> - **Visualization:** Generate heatmaps and CSV summaries of time spent looking at each screen region.
> - **Cross–Platform Support:** Runs on both macOS and Windows (CPU–only supported).
> - **Optional Head–Pose Estimation:** Includes head roll estimation using facial landmarks.

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Data Preparation](#data-preparation)
3. [DeepLabCut Model Training](#deeplabcut-model-training)
4. [Pipeline Overview](#pipeline-overview)
5. [Video Processing (Extract Landmarks)](#video-processing-extract-landmarks)
6. [Mapping Eye Landmarks to Screen Coordinates](#mapping-eye-landmarks-to-screen-coordinates)
7. [Visualization (Heatmaps and Section Durations)](#visualization-heatmaps-and-section-durations)
8. [Optional: Training a Gaze Mapping Model](#optional-training-a-gaze-mapping-model)
9. [Troubleshooting](#troubleshooting)
10. [Future Improvements](#future-improvements)

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

2. **(Optional) Touch Event Files**  
   If you are using additional calibration data from touches or clicks, store them similarly (e.g., `1.csv`, `2.csv`) with columns like `timestamp, x, y`. This step is only necessary if you intend to do a separate homography calibration or advanced mapping.

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

Below is the general workflow for analyzing a video and generating a gaze heatmap:

1. **`process_video.py`** – Takes an input video and uses DLC to extract eye landmarks (pupils/corners). Outputs a CSV file with the raw landmark coordinates and timestamps.
2. **`analyze_gaze.py`** – Takes the landmarks CSV plus a trained regression model, maps the landmarks to screen coordinates, splits the screen into sections, and outputs:
   - A heatmap image showing how much time was spent looking at each screen region.
   - A CSV summarizing time spent per region.
3. **(Optional)** **`train_gaze_mapping.py`** – If you need to train a new regression model that converts eye landmarks into screen coordinates, run this script with your calibration/training data.

---

## 5. Video Processing (Extract Landmarks)

Use **`process_video.py`** to process your video:

```bash
python process_video.py \
    --video path/to/video.mp4 \
    --config path/to/dlc_config.yaml \
    --output landmarks_output.csv
```

- **`--video`**: Path to your input video.  
- **`--config`**: Path to the DLC `config.yaml` file (from your trained project).  
- **`--output`**: Desired path for the output CSV of raw landmarks.

This script calls **`detect_eye.py`**, which:
- Loads each frame
- Temporarily writes it to a mini video
- Runs DeepLabCut analysis to detect `left_pupil`, `right_pupil`, `corner_left`, and `corner_right`
- (Optionally) computes head roll if you have a camera matrix and want that data
- Returns a CSV with columns like `left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, corner_left_x, corner_left_y, corner_right_x, corner_right_y, roll_angle, time`, etc.

---

## 6. Mapping Eye Landmarks to Screen Coordinates

After extracting the raw landmarks, you can map them to actual screen coordinates (e.g., if your screen is 1920×1080). This step requires a **trained regression model** or other mapping approach. By default, we provide a sample approach in **`train_gaze_mapping.py`** (see [Optional: Training a Gaze Mapping Model](#optional-training-a-gaze-mapping-model)).

---

## 7. Visualization (Heatmaps and Section Durations)

Use **`analyze_gaze.py`** to convert raw landmark CSVs into a heatmap and CSV summary of time spent per screen region:

```bash
python analyze_gaze.py \
    --landmarks_csv landmarks_output.csv \
    --model gaze_mapping_model.pkl \
    --screen_width 1920 \
    --screen_height 1080 \
    --n_cols 3 \
    --n_rows 3 \
    --output_heatmap gaze_heatmap.png \
    --output_sections section_durations.csv
```

- **`--landmarks_csv`**: The CSV from `process_video.py`.  
- **`--model`**: A pickle file containing your trained regression model (see below).  
- **`--screen_width`** / **`--screen_height`**: Dimensions of your screen or area of interest.  
- **`--n_cols`** / **`--n_rows`**: How many columns/rows to split the screen into.  
- **`--output_heatmap`**: Path to save the generated heatmap image.  
- **`--output_sections`**: Path to save a CSV summarizing how long (in seconds) you spent in each screen section.

This script:
1. Loads the CSV of raw landmarks.
2. Applies the trained model to each frame to get `(screen_x, screen_y)` gaze points.
3. Splits the screen into a grid (`section_mapping.py`).
4. Aggregates how many frames (or seconds) were spent in each region.
5. Saves a heatmap and a CSV listing time spent in each region.

---

## 8. Optional: Training a Gaze Mapping Model

If you don’t already have a regression model that maps `(pupil_x, pupil_y, corners, etc.)` to screen coordinates, you can create one with **`train_gaze_mapping.py`**:

```bash
python train_gaze_mapping.py \
    --data path/to/training_data.csv \
    --output data/trained_model/gaze_mapping_model.pkl
```

- Your CSV should include columns for the raw eye landmarks plus the known screen coordinates at each sample (i.e., ground truth).
- Once trained, you can reference the resulting `gaze_mapping_model.pkl` in `analyze_gaze.py`.

---

## 9. Troubleshooting

- **Model Issues / DLC Not Detecting Landmarks Properly:**
  - Verify that your `config.yaml` is correct and your DLC project is fully trained.
- **Mapping Accuracy Problems:**
  - Check that the data used to train your regression model covers a sufficient range of head poses and gaze positions.
- **Dependency Issues:**
  - Ensure all package versions match those listed in `requirements.txt`.
- **TensorFlow or Keras Errors:**
  - If you see `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'`, upgrade TensorFlow as shown in [Installation and Setup](#installation-and-setup).

---

## 10. Future Improvements

- **Enhanced Head–Pose Estimation:**
  - Refine the head pose calculations or incorporate more 3D facial landmarks to improve accuracy.
- **Adaptive Gaze Mapping:**
  - Implement an online or continuously updated calibration procedure.
- **Advanced Visualization Tools:**
  - Develop dynamic or interactive dashboards to visualize fixations and heatmaps in real time.

---

**Happy tracking!**  
