# Rhesus Macaque Gaze Tracker with Touch-Integrated Training (SLEAP Edition)

This repository implements an eye-tracking system for Rhesus macaques interacting with a touchscreen. In this edition, we use **SLEAP** (Social LEAP Estimates Animal Poses) for eye landmark detection instead of DeepLabCut (DLC). SLEAP offers faster training, high accuracy, and—with the proper setup—even near–real-time processing on a CPU-only system.

> **Key Features:**
> - **Offline Video Processing:** Analyze pre-recorded trial videos.
> - **SLEAP Integration:** Use a trained SLEAP model for eye landmark detection.
> - **Touch Event Correlation:** Align touch event logs (CSV/TXT) with gaze data.
> - **Gaze Mapping & Calibration:** Compute a calibration (homography) matrix that maps raw eye coordinates to screen coordinates and splits the screen into multiple small regions to determine fixation times.
> - **Optional Gaze Mapping Training:** Fine-tune the mapping model using paired data.
> - **Visualization:** Generate heatmaps or other summaries of fixation data.
> - **Cross–Platform:** Designed for macOS and Windows (optimized for CPU-only use).

---

## Usage Instructions

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/ahrebel/rhesustracking.git
cd rhesustracking
```

### 2. Create and Activate Your Python Environment

#### Using Conda (Recommended):

```bash
conda create -n monkey-gaze-tracker -c conda-forge python=3.7.12 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
```

#### Alternatively, Using pip with a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

Install the necessary Python dependencies:

```bash
pip install -r requirements.txt
```


### 4. Prepare Your Data

- **Video Files:**  
  Place your trial video files (e.g., `1.mp4`, `2.mp4`, etc.) into the `videos/input/` directory.

- **Touch Event Files:**  
  For each video, create a corresponding touch event file (e.g., `1.txt`) in CSV format with a header:
  ```
  timestamp,x,y
  ```
  Sample content:
  ```
  2025-02-24T18:41:57.2864969-05:00,16,15
  2025-02-24T18:41:58.6795674-05:00,34,25
  ```
  Ensure that timestamps follow ISO 8601 format.

### 5. Train Your SLEAP Model

Since accurate eye landmark detection is crucial, you must train a SLEAP model:
- **Label Frames:**  
  Launch the SLEAP labeling tool:
  ```bash
  sleap-label
  ```
  Create a new project and manually label the eye landmarks on a set of representative frames.

- **Train the Model:**  
  Use the SLEAP training command or GUI:
  ```bash
  sleap-train --config path/to/your_config.yaml
  ```
  (Consult the [SLEAP documentation](https://sleap.ai/installation) for detailed training instructions.)

- **Export the Model:**  
  Once training is complete, export your trained model (e.g., as a ZIP file) and place it in the `models/` directory. Then update the configuration in the repository (refer to the instructions in step 7) to point to your model file.

### 6. Update the Eye Detection Function

Before SLEAP integration, the system used a placeholder that returned the frame center as the eye coordinate. Now that you have a trained SLEAP model, update your eye detection function in the repository (for example, in a module like `src/detect_eye.py`) so that it:
- Loads your SLEAP model from the `models/` directory.
- Processes each video frame to detect the eye landmark(s).
- Returns the coordinates (e.g., the pupil or center of the eye).

*No additional changes to the calibration or analysis scripts are needed once this function is updated.*

### 7. Calibration

Run the calibration script to compute the homography (calibration) matrix. This script uses the first four touch events from each touch event file to compute the mapping from raw eye coordinates to screen coordinates.

```bash
python src/calibrate.py
```

Calibration files (e.g., `calibration_matrix_1.npy` and `calibration_1.yaml`) will be saved in the `data/trained_model/` folder.

### 8. Analyze Videos

Once calibration is complete and your SLEAP-based eye detection is working, run the analysis script. This script:
- Loads each video (and corresponding touch event file, if available).
- Uses the updated eye detection function (with SLEAP) to extract eye coordinates.
- Applies the calibration matrix to map these coordinates to screen coordinates.
- Divides the screen into predefined regions (for example, 110 sections) and computes the total time spent fixating in each region.
- Saves the output results (e.g., CSV and JSON) to the `data/analysis_output/` directory.

Run the analysis with:

```bash
python src/analyze_video.py
```

### 9. (Optional) Fine-Tune the Gaze Mapping

If you have additional paired training data (with columns such as `raw_x, raw_y, touch_x, touch_y`), you can improve the gaze mapping by running:

```bash
python src/train_gaze_mapping.py --data path/to/your_training_data.csv
```

This will update the calibration (gaze mapping) model and save the refined matrix to `data/trained_model/`.

### 10. (Optional) Visualize Gaze Data

To generate heatmaps or other visual summaries of the fixation data, run the visualization script:

```bash
python src/visualize.py --csv path/to/your_gaze_data.csv
```

### 11. Using the System for Screen Region Analysis

For your goal of splitting the screen into many small sections and determining the fixation time for each:
- The analysis script automatically divides the screen into a grid (e.g., 110 regions) based on the video frame dimensions.
- As the video is processed, the system maps each gaze point to one of these regions.
- The cumulative duration for which the gaze falls within each region is computed.
- Results are saved in a file (e.g., CSV or JSON) detailing the total fixation time per section.

### Final Verification

After installation and setup, run the following command to verify that everything is correctly installed:

```bash
python -c "import tables, torch, sleap; print('Installation successful')"
```

You should see:

```
Installation successful
```

---

## Troubleshooting

- **Model Loading Errors:**  
  Ensure that the SLEAP model path is correctly specified in your configuration.
- **No Touch File Found:**  
  Verify that your touch event files share the same base name as the video files and are located in the `videos/input/` directory.
- **Performance Issues:**  
  SLEAP is optimized for GPU but works on CPU. If processing is slow, consider reducing video resolution.

Happy tracking!
