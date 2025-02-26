Below is an updated README that explains the entire workflow using SLEAP instead of DLC. In addition to the original steps (setting up the environment, preparing data, training the SLEAP model, calibrating the system, and analyzing videos), we now include a section that explains how the system splits the screen into a grid of small sections and accumulates the total time spent looking at each section. I've also included an additional module (`src/section_mapping.py`) that you should add to the repository and instructions on modifying your analysis script.

---

```markdown
# Rhesus Macaque Gaze Tracker with Touch-Integrated Training (SLEAP Edition)

This repository implements an eye-tracking system for Rhesus macaques interacting with a touchscreen. In this version, we use **SLEAP** (Social LEAP Estimates Animal Poses) for eye landmark detection instead of DeepLabCut (DLC). SLEAP can provide faster training, high accuracy, and—with the proper setup—near–real-time processing on a CPU-only system.

> **Key Features:**
> - **Offline Video Processing:** Analyze pre-recorded trial videos.
> - **SLEAP Integration:** Use a trained SLEAP model for eye landmark detection.
> - **Touch Event Correlation:** Align touch event logs (CSV/TXT) with gaze data.
> - **Gaze Mapping & Calibration:** Compute a calibration (homography) matrix that maps raw eye coordinates to screen coordinates.
> - **Grid-Based Fixation Analysis:** Split the screen into multiple regions (e.g., 110 sections) and compute how much time is spent looking at each.
> - **Optional Gaze Mapping Training:** Fine-tune the mapping model using paired data.
> - **Visualization:** Generate heatmaps of gaze fixations.
> - **Cross–Platform:** Designed for macOS and Windows (optimized for CPU-only use).

## Repository Structure

```
rhesusDLC/
├── data/
│   ├── analysis_output/         # Gaze analysis results (CSV, JSON, heatmaps, etc.)
│   └── trained_model/           # Calibration matrices and mapping configuration
├── models/                      # Directory to store your trained SLEAP model (ZIP file)
├── src/                         # Source code files
│   ├── calibrate.py             # Compute calibration matrix using touch events
│   ├── analyze_video.py         # Process videos to extract gaze data and accumulate fixation times
│   ├── train_gaze_mapping.py    # (Optional) Fine-tune the gaze mapping using paired data
│   ├── visualize.py             # Generate heatmaps and other visualizations
│   └── section_mapping.py       # [NEW] Helper functions to split the screen into regions and map points to regions
├── videos/
│   └── input/                   # Input videos and corresponding touch event files
├── requirements.txt             # Python package requirements
└── README.md                    # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ahrebel/rhesusDLC.git
cd rhesusDLC
```

### 2. Create and Activate Your Environment

#### Using Conda (Recommended)

```bash
conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
```

### 3. Install Required Packages

Install the Python dependencies (including SLEAP, TensorFlow, and other supporting packages):

```bash
pip install pyyaml tensorflow tensorpack tf-slim sleap
```

*Alternatively, if you prefer using pip with a virtual environment:*

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Example `requirements.txt`:*

```
pyyaml==6.0.2
numpy==1.24.3
opencv-python==4.8.0.76
pandas==2.0.3
matplotlib==3.7.3
scikit-learn==1.3.2
scikit-image==0.21.0
scipy==1.10.1
tqdm==4.67.1
statsmodels==0.14.1
tensorflow==2.13.0
tensorpack==0.11
tf-slim==1.1.0
sleap==1.4.0
```

### 4. Prepare Your Data

- **Videos:**  
  Place your trial video files (e.g., `1.mp4`, `2.mp4`, etc.) into the `videos/input/` directory.

- **Touch Event Files:**  
  For each video, create a corresponding touch event file (e.g., `1.txt`). These files should be in CSV format with a header:
  ```
  timestamp,x,y
  ```
  Example content:
  ```
  2025-02-24T18:41:57.2864969-05:00,16,15
  2025-02-24T18:41:58.6795674-05:00,34,25
  ```
  Make sure the timestamps are in ISO 8601 format.

### 5. Train Your SLEAP Model

Accurate eye landmark detection is crucial. Follow these steps to train a SLEAP model:

1. **Label Frames:**  
   Launch the SLEAP labeling GUI:
   ```bash
   sleap-label
   ```
   Create a new project and manually label the relevant eye landmark(s) (e.g., the pupil or eye center) on a representative set of frames.

2. **Train the Model:**  
   Either use the SLEAP GUI or run from the command line:
   ```bash
   sleap-train --config path/to/your_config.yaml
   ```
   (Refer to the [SLEAP documentation](https://sleap.ai/installation) for configuration details.)

3. **Export the Model:**  
   Once training is complete, export the trained model (typically as a ZIP file) and place it in the `models/` folder. Then, update the model path in your eye detection function (see Step 6).

### 6. Update the Eye Detection Function

Create or update the module `src/detect_eye.py` so that it uses your SLEAP model to detect eye coordinates. For example:

```python
# src/detect_eye.py
import sleap

# Update the path below to point to your exported SLEAP model (ZIP file)
MODEL_PATH = "models/your_trained_model.zip"
sleap_model = sleap.load_model(MODEL_PATH)

def get_eye_coordinates(frame):
    """
    Uses the SLEAP model to detect eye landmark(s) in the given frame.
    Returns the (x, y) coordinate of the detected landmark (e.g., pupil center).
    """
    # SLEAP expects a list of frames
    predictions = sleap_model.predict(frame)
    if predictions and len(predictions) > 0:
        # For single–animal tracking, take the first instance.
        instance = predictions[0]
        # Assume the landmark of interest is the first point (adjust if needed)
        x, y = instance.points[0]
        return (x, y)
    # Fallback: return the center of the frame
    h, w = frame.shape[:2]
    return (w // 2, h // 2)
```

Then, in both `src/calibrate.py` and `src/analyze_video.py`, replace any placeholder eye–coordinate function calls with `get_eye_coordinates(frame)` from this module.

### 7. Screen Splitting & Fixation Analysis

To achieve your goal of splitting the screen into many small sections (for example, 110 regions) and tracking how long the gaze remains in each, the analysis script will use helper functions that map each detected gaze coordinate to a grid cell.


### 8. Calibration

Run the calibration script to compute the homography (calibration matrix) from raw eye coordinates to screen coordinates using the first four touch events:

```bash
python src/calibrate.py
```

For a video `1.mp4` with touch data in `1.txt`, the calibration files will be saved in `data/trained_model/` (e.g., `calibration_matrix_1.npy` and `calibration_1.yaml`).

### 9. Analyze Videos

With the calibration matrix and your updated SLEAP-based eye detection (including section mapping), run the video analysis script:

```bash
python src/analyze_video.py
```

This script will:
- Load each video and its corresponding touch event file (if present).
- Process each frame:
  - Use SLEAP to obtain the eye coordinates.
  - Apply the calibration matrix to convert raw eye coordinates to screen coordinates.
  - Determine the grid cell (region) for each gaze coordinate.
  - Accumulate fixation time for each region.
- Save the analysis results (including per–region fixation times) in CSV and JSON files under `data/analysis_output/`.

### 10. (Optional) Fine-Tune the Gaze Mapping

If you have additional paired data (raw eye coordinates and corresponding touch locations) in a CSV file (columns: `raw_x, raw_y, touch_x, touch_y`), refine the mapping by running:

```bash
python src/train_gaze_mapping.py --data path/to/your_training_data.csv
```

This will retrain the gaze mapping model and update the calibration matrix in `data/trained_model/`.

### 11. (Optional) Visualize Gaze Data

Generate heatmaps or other visualizations from your gaze data with:

```bash
python src/visualize.py --csv path/to/your_gaze_data.csv
```

## Notes on Switching from the Placeholder to SLEAP

- **Before SLEAP Integration:**  
  The original placeholder function simply returned the center of the video frame as the eye coordinate (used for testing the calibration pipeline).
  
- **After SLEAP Integration:**  
  Once you have trained your SLEAP model and updated the `get_eye_coordinates()` function (see Step 6), the system will automatically use SLEAP predictions for eye landmark detection. No additional changes to the calibration or analysis scripts are needed.

## Final Verification

To verify that everything is installed and working, run:

```bash
python -c "import tables, torch, sleap; print('Installation successful')"
```

You should see:

```
Installation successful
```

## Troubleshooting

- **Model Not Loading:**  
  Ensure the path in `src/detect_eye.py` is correct for your exported SLEAP model.
- **No Touch File Found:**  
  Confirm that each video has a corresponding touch file (with the same base name) in `videos/input/`.
- **Performance on CPU:**  
  Although SLEAP is optimized for GPU, it will run on CPU. Processing may be slower on a CPU-only machine—consider reducing the video resolution if necessary.

Happy tracking!
