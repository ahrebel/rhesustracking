# Rhesus Macaque Gaze Tracker (DeepLabCut)

![Animated Eyes](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye–tracking system for Rhesus macaques and humans by using **DeepLabCut (DLC)**. The system robustly extracts eye (and head) landmarks from video frames, maps raw eye coordinates to screen coordinates via calibration, and analyzes gaze and fixation patterns during touchscreen interactions.

> **Key Features:**
> - **Offline Video Processing:** Process pre–recorded trial videos for analysis.
> - **DeepLabCut-Based Landmark Detection:** Uses a trained DLC model to detect eye landmarks.
> - **Calibration & Gaze Mapping:** Computes a calibration (homography) matrix to accurately map raw eye coordinates to screen coordinates and trains a regression model to further refine gaze mapping.
> - **Visualization:** Generate plots, heatmaps, and other summaries to visualize gaze distribution.
> - **Cross–Platform:** Designed to run on macOS and Windows systems with CPU–only setups.
> - **Enhanced Head–Pose Estimation:** Incorporates head–pose (e.g., head roll) estimation using facial landmarks.

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
9. [Data Loading](#data-loading)
10. [Final Verification](#final-verification)
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

#### Additional Dependencies:

- **DeepLabCut:**
  ```bash
  pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
  ```

---

## 2. Data Preparation

### Video Files

Place your trial videos (e.g., `1.mp4`, `2.mp4`, etc.) in a folder such as `videos/input/`.

### Touch Event Files

For each video, create a corresponding touch event file (e.g., `1.csv`) formatted as CSV with a header:
```csv
timestamp,x,y
```
Example:
```csv
timestamp,x,y
2025-02-24T18:41:57.2864969-05:00,16,15
2025-02-24T18:41:58.6795674-05:00,34,25
```
Ensure timestamps use ISO 8601 format.

---

## 3. DeepLabCut Model Training

1. **Launch DLC GUI:**
   ```bash
   python -m deeplabcut
   ```
2. **Create a New Project:**  
   - Enter your project name and add the video(s).
   - Label keypoints such as `left_pupil`, `right_pupil`, `corner_left`, and `corner_right`.
3. **Label Frames:**  
   Use a diverse set of frames.
4. **Train the Network:**  
   Use the **"Train Network"** option. For CPU-only systems, consider a lighter network (e.g., `mobilenet_v2_1.0`).
5. **Evaluate the Model:**  
   Use **"Evaluate Network"** to assess accuracy.
6. **Update Config Path:**  
   In `src/detect_eye_dlc.py`, set the `PROJECT_CONFIG` variable to your DLC project’s `config.yaml`.

---

## 4. Eye Detection Integration

- **DeepLabCut Detection:**  
  The detection code in `src/detect_eye_dlc.py` uses DLC to process a frame, extract the four key landmarks, compute the average eye coordinate, and estimate head roll.

- **Unified Interface:**  
  `src/detect_eye.py` simply aliases the DLC detector.

---

## 5. Calibration

Run the calibration script to compute a homography matrix mapping raw eye coordinates to screen coordinates:
```bash
python src/calibrate.py
```
The script uses touch event data and sample eye points, saving the calibration files in `data/trained_model/`.

---

## 6. Video Analysis

Analyze videos using:
```bash
python src/analyze_video.py --video path/to/video.mp4 --config path/to/your/config.yaml --output path/to/output.csv
```
Each frame is processed with DLC detection, and the results are saved to a CSV.

---

## 7. Optional: Fine–Tuning Gaze Mapping

If you have paired data, further refine the mapping by running:
```bash
python src/train_gaze_mapping.py --data path/to/your_training_data.csv
```
This trains a regression model (saved as a pickle file) that maps raw eye features to screen coordinates.

---

## 8. Visualization

Generate visual summaries by running:
```bash
python src/visualize.py --csv path/to/your_gaze_data.csv
```
This creates heatmaps, fixation plots, etc.

---

## 9. Data Loading

Use `data_loader.py` to load gaze and touch event data for further analysis.

---

## 10. Final Verification

Verify installations:
```bash
python -c "import tables, deeplabcut; print('Installation successful')"
```
You should see:
```
Installation successful
```

---

## 11. Troubleshooting

- **Model Issues:**  
  Ensure your DLC model path and config are correct.
- **Calibration Errors:**  
  Verify touch event file formats.
- **Performance:**  
  Consider reducing video resolution or using lighter models.
- **Dependencies:**  
  If issues arise, check your package versions against the requirements.

---

## 12. Future Improvements

- **Enhanced Head–Pose Estimation:**  
  Further refine head–pose calculations.
- **Learning-Based Gaze Mapping:**  
  Incorporate a continuous learning module.
- **Advanced Visualization:**  
  Expand visualization tools to include dynamic heatmaps and fixation analyses.

---

Happy tracking!
