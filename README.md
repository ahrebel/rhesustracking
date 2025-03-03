# Rhesus Macaque Gaze Tracker (DeepLabCut)

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye–tracking system for Rhesus macaques (and humans) using **DeepLabCut (DLC)**. The system detects eye and head landmarks from video frames, maps raw eye coordinates to screen positions via calibration, and analyzes gaze and fixation patterns during touchscreen interactions.

> **Key Features:**
>
> - **Offline Video Processing:** Analyze pre–recorded trial videos.
> - **DeepLabCut–Based Landmark Detection:** Use a trained DLC model to detect key eye landmarks.
> - **Calibration & Gaze Mapping:** Compute a homography matrix and refine mapping via a regression model to accurately translate eye coordinates to screen coordinates.
> - **Visualization:** Generate heatmaps and other visual summaries of gaze distribution.
> - **Cross–Platform Support:** Runs on both macOS and Windows (CPU–only supported).
> - **Enhanced Head–Pose Estimation:** Includes head roll estimation using facial landmarks.

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Data Preparation](#data-preparation)
3. [DeepLabCut Model Training](#deeplabcut-model-training)
4. [Eye Detection Integration](#eye-detection-integration)
5. [Calibration](#calibration)
6. [Video Analysis](#video-analysis)
7. [Optional: Fine–Tuning Gaze Mapping](#optional-fine-tuning-gaze-mapping)
8. [Visualization](#visualization)
9. [Data Loading and Final Verification](#data-loading-and-final-verification)
10. [Troubleshooting](#troubleshooting)
11. [Future Improvements](#future-improvements)

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

### Video Files

- Place your trial videos (e.g., `1.mp4`, `2.mp4`, etc.) in the `videos/input/` folder.

### Touch Event Files

- For each video, create a corresponding CSV file (e.g., `1.csv`) with the following header and format:
  
  ```csv
  timestamp,x,y
  2025-02-24T18:41:57.2864969-05:00,16,15
  2025-02-24T18:41:58.6795674-05:00,34,25
  ```
  
- Ensure timestamps are in ISO 8601 format.

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
   - In `src/detect_eye_dlc.py`, set the `PROJECT_CONFIG` variable to point to your project's `config.yaml`.

---

## 4. Eye Detection Integration

- **DeepLabCut Detection:**  
  The module in `src/detect_eye_dlc.py` processes each video frame with DLC to extract the four key landmarks, compute the eye's average coordinate, and estimate head roll.

- **Unified Interface:**  
  The file `src/detect_eye.py` provides a simple alias to the DLC detection routine.

---

## 5. Calibration

- Run the calibration script to compute the homography mapping from raw eye coordinates to screen coordinates.  
- **Command:**

  ```bash
  python src/calibrate.py
  ```

- Calibration uses touch event data and sample eye points, saving the resulting calibration files in `data/trained_model/`.

---

## 6. Video Analysis

- Process your trial videos to extract gaze data:
  
  ```bash
  python src/analyze_video.py --video path/to/video.mp4 --config path/to/your/config.yaml --output path/to/output.csv
  ```

- This script processes each frame with DLC, calculates timestamps based on video FPS, and outputs a CSV containing columns for frame number, timestamp, eye x/y coordinates, and head roll.

---

## 7. Optional: Fine–Tuning Gaze Mapping

- If you have paired calibration data (eye landmarks and click positions), refine the mapping by training a regression model:
  
  ```bash
  python src/train_gaze_mapping.py --data path/to/your_training_data.csv
  ```

- This step creates a continuous mapping from eye features to screen coordinates and saves the regression model (for example, as a pickle file).

---

## 8. Visualization

- Generate visual summaries (heatmaps, fixation plots, etc.) from your gaze data:

  ```bash
  python src/visualize.py --csv path/to/your_gaze_data.csv
  ```

---

## 9. Data Loading and Final Verification

- Use `data_loader.py` to import gaze and touch event data for further analysis.
- To verify that all required packages are installed correctly, run:

  ```bash
  python -c "import tables, deeplabcut; print('Installation successful')"
  ```

  You should see:
  ```
  Installation successful
  ```

---

## 10. Troubleshooting

- **Model Issues:**  
  Double-check your DLC model path and configuration file location.
- **Calibration Errors:**  
  Verify that your touch event files are properly formatted and that the timestamps match your video analysis output.
- **Performance Concerns:**  
  Reduce video resolution or consider using a lighter network for CPU-only setups.
- **Dependency Issues:**  
  Ensure all package versions meet the requirements listed in `requirements.txt`.
- **TensorFlow Errors:**  
  If you see `ModuleNotFoundError: No module named 'keras.legacy_tf_layers'`, update your TensorFlow installation as shown in the Installation section.

---

## 11. Future Improvements

- **Enhanced Head–Pose Estimation:**  
  Further refine head pose calculations to improve gaze mapping.
- **Learning-Based Gaze Mapping:**  
  Implement a continuously learning module for adaptive calibration.
- **Advanced Visualization Tools:**  
  Develop dynamic heatmaps and more detailed fixation analysis reports.

---

Happy tracking!

Feel free to open an issue or submit a pull request if you have suggestions or encounter any problems.
