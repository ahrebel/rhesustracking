# Rhesus Macaque Gaze Tracker (DeepLabCut & SLEAP)

![20250225_2228_Animated Eyes in Cyberspace_remix_01jn04sjzbeb2v028a2akx9nex](https://github.com/user-attachments/assets/0f245b14-ec20-4a11-868a-ae207a7dfa1d)

This repository implements an eye–tracking system for Rhesus macaques and humans by fusing landmark detection from **DeepLabCut (DLC)** and **SLEAP**. The system robustly extracts eye (and head) landmarks from video frames, maps raw eye coordinates to screen coordinates via calibration, and analyzes gaze and fixation patterns during touchscreen interactions.

> **Key Features:**
> - **Offline Video Processing:** Process pre–recorded trial videos for analysis.
> - **Dual-Model Integration:** Uses both a trained DLC model and a trained SLEAP model to detect eye landmarks.
> - **Fusion Pipeline:** Combines outputs from DLC and SLEAP using weighted averaging for enhanced robustness.
> - **Touch Event Correlation:** Synchronize touch event logs (CSV/TXT) with gaze data.
> - **Gaze Mapping & Calibration:** Compute a calibration (homography) matrix to accurately map raw eye coordinates to screen coordinates.
> - **Optional Gaze Mapping Training:** Fine–tune the calibration model with paired training data.
> - **Visualization:** Generate plots, heatmaps, and other summaries to visualize gaze distribution.
> - **Cross–Platform:** Designed to run on macOS and Windows systems with CPU–only setups.
> - **Enhanced Head–Pose Estimation:** Incorporates head–pose (e.g., head roll) estimation using facial landmarks.

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Data Preparation](#data-preparation)
3. [DeepLabCut Model Training](#deeplabcut-model-training)
4. [SLEAP Model Training](#sleap-model-training)
5. [Integration & Eye Detection](#integration--eye-detection)
6. [Calibration](#calibration)
7. [Video Analysis](#video-analysis)
8. [Optional: Fine–Tuning Gaze Mapping](#optional-fine-tuning-gaze-mapping)
9. [Visualization](#visualization)
10. [Data Loading](#data-loading)
11. [Final Verification](#final-verification)
12. [Troubleshooting](#troubleshooting)
13. [Future Improvements](#future-improvements)

---

## 1. Installation and Setup

### Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/ahrebel/rhesustracking.git
cd rhesustracking
```

### Create and Activate Your Python Environment

#### Using Conda (Recommended):

Make sure you have Miniconda installed:
[Download Anaconda/Miniconda](https://www.anaconda.com/download/)

Then run:

```bash
conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
```

#### Alternatively, Using pip with a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### Install Required Packages

Install all necessary packages:

```bash
pip install -r requirements.txt
```

#### Additional Dependencies:

- **DeepLabCut:**
  ```bash
  pip install deeplabcut pyyaml tensorflow tensorpack tf-slim 'deeplabcut[gui]'
  ```
- **SLEAP:**
  ```bash
  pip install sleap
  pip install cattrs python-rapidjson jsmin albumentations jsonpickle pykalman seaborn ndx_pose
  ```

> **Note:** DeepLabCut and SLEAP may install additional dependencies during their setup. If you experience issues, follow the instructions above or check the documentation for the software.

---

## 2. Data Preparation

### Video Files

- **Input Videos:**  
  Place your trial videos (e.g., `1.mp4`, `2.mp4`, etc.) in a designated folder (for example, `videos/input/`).

### Touch Event Files

- **Touch Logs:**  
  For each video, create a corresponding touch event file (e.g., `1.csv`) formatted as CSV with a header:
  ```csv
  timestamp,x,y
  ```
  **Sample Content:**
  ```csv
  2025-02-24T18:41:57.2864969-05:00,16,15
  2025-02-24T18:41:58.6795674-05:00,34,25
  ```
  Ensure timestamps are in ISO 8601 format.

---

## 3. DeepLabCut Model Training

Accurate eye landmark detection with DLC is critical. Follow these steps:

1. **Launch the DLC GUI:**
   ```bash
   python -m deeplabcut
   ```
2. **Create a New Project:**  
   - Enter your project name and add the video(s).
   - Label keypoints such as `left_pupil`, `right_pupil`, `corner_left`, and `corner_right` (add more if needed for head–pose estimation).
3. **Label Frames:**  
   Use a diverse set of frames to ensure robust training.
4. **Train the Network:**  
   Use the **"Train Network"** option. For CPU-only systems, consider using a lighter model (e.g., `mobilenet_v2_1.0`).
5. **Evaluate the Model:**  
   Use the **"Evaluate Network"** option to assess accuracy and refine if necessary.
6. **Update Config Path:**  
   In `src/detect_eye_dlc.py`, ensure the variable `PROJECT_CONFIG` points to your DLC project’s `config.yaml` file.  
   *Example:*
   ```python
   PROJECT_CONFIG = '/Users/anthonyrebello/rhesustracking/eyetracking-ahrebel-2025-02-26/config.yaml'
   ```

---

## 4. SLEAP Model Training

Training a SLEAP model complements DLC by providing additional landmark detection capabilities. Follow these steps to train a SLEAP model for both Rhesus macaques and humans:

1. **Install SLEAP:**
   Ensure SLEAP is installed:
   ```bash
   pip install sleap
   ```
2. **Launch the SLEAP Labeling GUI:**
   ```bash
   sleap-label
   ```
   - This opens the SLEAP GUI.
3. **Create a New SLEAP Project:**
   - Import your video(s) or images into SLEAP.
   - Label keypoints for each subject. **Essential labels:** `left_eye` and `right_eye`. You may add more keypoints as needed.
4. **Label a Diverse Set of Frames:**
   - Ensure your training dataset includes frames under varying conditions (lighting, pose, etc.) for both macaques and humans.
5. **Train the Model:**
   - Once labeling is complete, use the SLEAP training tool:
     ```bash
     sleap-train config.yaml
     ```
     - SLEAP will generate a training configuration file (usually named `config.yaml`). Adjust training parameters if necessary.
   - Monitor training progress until convergence.
6. **Export the Trained Model:**
   - After training, export the model checkpoint (e.g., `sleap_model.ckpt`).
7. **Update Model Path:**
   - In `src/detect_eye_sleap.py`, set the `MODEL_PATH` variable to the location of your trained SLEAP checkpoint.
   *Example:*
   ```python
   MODEL_PATH = 'models/sleap_model.ckpt'
   ```

For more detailed instructions, refer to the [SLEAP GitHub repository](https://github.com/talmo/sleap) and its documentation.

---

## 5. Integration & Eye Detection

The pipeline now leverages both models:
- **DLC Detection:** Your working DLC code is located in `src/detect_eye_dlc.py`.
- **SLEAP Detection:** New SLEAP detection is implemented in `src/detect_eye_sleap.py`.
- **Fusion Module:** In `src/fuse_landmarks.py`, outputs from both models are combined via weighted averaging.

The main detection function in `src/detect_eye.py` calls both detectors and fuses their outputs. This hybrid approach increases robustness, especially in challenging scenarios.

---

## 6. Calibration

### Running Calibration

Map raw (fused) eye coordinates to screen coordinates using your touch event data:
1. **Run the Calibration Script:**
   ```bash
   python src/calibrate.py
   ```
2. **Processing Details:**  
   - The script reads the first four touch events from each event file.
   - It computes a homography matrix mapping raw eye coordinates to screen coordinates.
3. **Output:**  
   Calibration files (e.g., `calibration_matrix_1.npy` and `calibration_1.yaml`) are saved in the `data/trained_model/` folder.

---

## 7. Video Analysis

After calibration, analyze your videos:

1. **Run the Analysis Script:**
   ```bash
   python src/analyze_video.py --video path/to/video.mp4 --output path/to/output.csv
   ```
2. **Processing Details:**  
   - Each frame is processed using the fused detection function.
   - The system extracts raw eye coordinates, head pose (roll angle), and applies the calibration matrix to map them to screen coordinates.
3. **Output:**  
   Results are saved as a CSV file (including frame number, eye coordinates, and head pose parameters).

---

## 8. Optional: Fine–Tuning Gaze Mapping

If you have additional paired data (raw coordinates and click positions), further refine the mapping:
1. **Run the Gaze Mapping Training Script:**
   ```bash
   python src/train_gaze_mapping.py --data path/to/your_training_data.csv
   ```
2. **Result:**  
   This updates the calibration model and saves a refined calibration matrix.

---

## 9. Visualization

Generate visual summaries of gaze data:
1. **Run the Visualization Script:**
   ```bash
   python src/visualize.py --csv path/to/your_gaze_data.csv
   ```
2. **Output:**  
   Heatmaps, fixation plots, and other visualizations are produced to help interpret the data.

---

## 10. Data Loading

Use `data_loader.py` to load gaze and touch event data for further analysis:

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

---

## 11. Final Verification

Verify that all components are installed correctly:

```bash
python -c "import tables, deeplabcut; print('Installation successful')"
```

Expected output:

```
Installation successful
```

---

## 12. Troubleshooting

- **DeepLabCut Model Issues:**  
  Ensure that the DLC model path in your configuration is correct and that the model has been properly trained.
- **SLEAP Model Issues:**  
  Confirm that your SLEAP checkpoint path (`MODEL_PATH` in `src/detect_eye_sleap.py`) is correct and that the model is trained on both species.
- **Calibration Errors:**  
  Verify that your touch event files are correctly formatted and share the same base name as their corresponding video files.
- **Performance Problems:**  
  For CPU-only systems, consider reducing video resolution or using lighter network architectures.
- **Dependency Conflicts:**  
  Check package versions in `requirements.txt` and install DLC and SLEAP last to allow them to resolve dependencies.

---

## 13. Future Improvements

- **Enhanced Head–Pose Robustness:**  
  Further refine the 3D facial model and incorporate additional facial landmarks.
- **Learning-Based Gaze Mapping:**  
  Integrate a learning module to continuously improve the mapping model with new data.
- **Extended Visualization:**  
  Expand visualization tools to include dynamic heatmaps, fixation duration histograms, and region-based analysis.

---

Happy tracking!

If you encounter any issues or have suggestions for further improvements, please open an issue on GitHub.
