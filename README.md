
# Rhesus Macaque Gaze Tracker with Touch-Integrated Training



**Note:** This repository is a work in progress. The instructions below may need to be altered depending on your data, hardware, and goals. 
The system uses DeepLabCut (DLC) for detecting eye landmarks, calibrates the mapping between raw eye coordinates and screen regions, and can correlate touchscreen events with gaze data.



There are two main modes in this project:

1. **Analysis Mode:**  
   Processes trial videos to calculate the total fixation time in each of 110 predefined screen sections. If a corresponding touch event file is available, it also aligns touch data with the gaze information.

2. **Training Mode:**  
   Uses paired data (raw eye coordinates and touch locations) to train a regression (or homography) model that refines the mapping from eye coordinates to screen locations.

**Important:** Training (or fine-tuning) the DeepLabCut model for eye landmark detection is separate from calibrating and training the gaze mapping model. (The calibration and analysis steps assume that you have a DLC model in place that produces reliable eye coordinates.) In the default code, a placeholder function returns the center of the frame. You must switch to your trained DLC model when ready.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Initial Calibration and Analysis](#initial-calibration-and-analysis)
4. [Training the Gaze Mapping Model (Optional)](#training-the-gaze-mapping-model-optional)
5. [Switching from Placeholder to DLC](#switching-from-placeholder-to-dlc)
6. [Training the DeepLabCut (DLC) Model](#training-the-deeplabcut-dlc-model)
7. [Final Verification](#final-verification)

---

## 1. Environment Setup

### Clone the Repository

```bash
git clone https://github.com/ahrebel/rhesusDLC
cd rhesusDLC
```

### Set Up Your Python Environment

#### Using Conda (Recommended):

```bash
conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
```

Then install the required Python packages:

```bash
pip install pyyaml tensorflow tensorpack tf-slim
pip install "deeplabcut[gui,tf]"
```

> **Note:** If you prefer to use a virtualenv with pip only, create and activate your virtual environment and then run:
>
> ```bash
> pip install -r requirements.txt
> ```

---

## 2. Data Preparation

### Video Files

Place your trial video files (e.g., `1.mp4`, `trial1.mp4`, etc.) in the `videos/input/` directory.

### Touch Event Files

For each video, provide a corresponding touch event file (e.g., `1.txt` or `1_touch.csv`) with matching base names. Each file should have a header and rows with the following columns:

```
timestamp,x,y
2025-02-24T18:41:57.2864969-05:00,16,15
2025-02-24T18:41:58.6795674-05:00,34,25
...
```

Make sure timestamps are in ISO 8601 format.

---

## 3. Initial Calibration and Analysis

### A. Calibration

Run the calibration script to compute a homography matrix that maps raw eye coordinates to screen coordinates (using the first four touch events):

```bash
python src/calibrate.py
```

For example, for `1.mp4` with `1.txt`, calibration files will be saved as:
- `data/trained_model/calibration_matrix_1.npy`
- `data/trained_model/calibration_1.yaml`

### B. Analyze Videos

Once calibrated, process your videos with:

```bash
python src/analyze_video.py
```

This script will:
- Read the video,
- Extract eye coordinates (by default using a placeholder that returns the center of the frame),
- Apply the calibration matrix,
- Map gaze to screen sections,
- Save fixation results (CSV/JSON), and
- If a corresponding touch file is found, align touch events with gaze data.

*Note:* At this point the system is still using a placeholder for eye landmark extraction.

---

## 4. Train the Gaze Mapping Model (Optional)

If you have a CSV file containing paired data (columns: `raw_x, raw_y, touch_x, touch_y`), update your calibration (gaze mapping) by running:

```bash
python src/train_gaze_mapping.py --data path/to/your_training_data.csv
```

This will generate a new calibration matrix saved in the `data/trained_model/` directory.

---

## 5. Switching from Placeholder to DLC

By default, the analysis script uses a placeholder function for eye coordinate extraction. To use your trained DeepLabCut model instead:

1. **Train a DLC Model** (see Section 6 below) and obtain its configuration file (e.g. `config.yaml`).

2. **Modify the Analysis Script:**  
   In `src/analyze_video.py`, replace the placeholder function with code that loads and uses the DLC model. For example, change:

   ```python
   # Placeholder:
   def get_eye_coordinates(frame):
       h, w = frame.shape[:2]
       return (w/2, h/2)
   ```

   to

   ```python
   from deeplabcut.pose_estimation_tensorflow.predictor import Predictor
   # Initialize your DLC predictor with your DLC config file:
   predictor = Predictor('/path/to/your/DLC/config.yaml', shuffle=1)
   
   def get_eye_coordinates(frame):
       # Replace with the actual method to get eye landmarks
       return predictor.predict(frame)
   ```

3. **Test the Integration:**  
   Run the analysis script again:

   ```bash
   python src/analyze_video.py
   ```

   Verify that the model now returns DLC predictions instead of the placeholder output.

---

## 6. Training the DeepLabCut (DLC) Model

**Note:** Training DLC is independent of the gaze mapping calibration. Follow these steps if you need to train or fine-tune a DLC model for eye landmark detection:

### A. Create or Use an Existing DLC Project

If you don’t have a DLC project yet, create one (refer to the [DeepLabCut documentation](https://deeplabcut.github.io/DeepLabCut/docs/intro.html)):

```bash
python -m deeplabcut.create_project --project MonkeyGaze --experimenter YourName --videos /path/to/your_video.mp4 --working_directory /path/to/your_project_folder
```

*Note:* DLC v2.3.11 may have slight differences in the command-line interface.

### B. Extract Frames for Labeling

From your DLC project folder (where `config.yaml` resides):

```bash
python -m deeplabcut.extract_frames /path/to/config.yaml --algo 2
```

This extracts a set of frames for manual labeling.

### C. Label the Frames

Run the DLC labeling GUI:

```bash
python -m deeplabcut.label_frames /path/to/config.yaml
```

Label the desired eye landmarks on each frame and save your annotations.

### D. Create the Training Dataset

After labeling, create the training dataset:

```bash
python -m deeplabcut.create_training_dataset /path/to/config.yaml --shuffle 1
```

### E. Train the Network

Start training your DLC model:

```bash
python -m deeplabcut.train_network /path/to/config.yaml --shuffle 1
```

Monitor training loss and adjust parameters in `config.yaml` as needed.

### F. Evaluate the Model

After training, evaluate performance with:

```bash
python -m deeplabcut.evaluate_network /path/to/config.yaml --shuffle 1
```

### G. Analyze New Videos with DLC

When satisfied with the model’s performance, analyze new videos:

```bash
python -m deeplabcut.analyze_videos /path/to/config.yaml --videos /path/to/new_video.mp4
```

For more details, consult the [DeepLabCut documentation](https://deeplabcut.github.io/DeepLabCut/docs/).

---

## 7. Final Verification

To confirm your installation is correct, run:

```bash
python -c "import tables, torch, deeplabcut; print('Installation successful')"
```

You should see output similar to:

```
Loading DLC 2.3.11...
DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)
Installation successful
```

---

Happy tracking!
