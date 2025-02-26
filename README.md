# Rhesus Macaque Gaze Tracker (DeepLabCut Edition)

This repository implements an eye–tracking system for Rhesus macaques interacting with a touchscreen using **DeepLabCut (DLC)** for eye–landmark detection. DeepLabCut provides a user–friendly GUI for labeling, training, and analysis. Once trained, the model is used to extract eye coordinates from video frames, which are then mapped to screen coordinates via calibration.

> **Key Features:**
> - **Offline Video Processing:** Analyze pre–recorded trial videos.
> - **DeepLabCut Integration:** Use a trained DLC model for eye landmark detection.
> - **Touch Event Correlation:** Align touch event logs (CSV/TXT) with gaze data.
> - **Gaze Mapping & Calibration:** Compute a calibration (homography) matrix to map raw eye coordinates to screen coordinates.
> - **Optional Gaze Mapping Training:** Fine–tune the mapping model using paired data.
> - **Visualization:** Generate plots or heatmaps of gaze points.
> - **Cross–Platform:** Designed for macOS and Windows (optimized for CPU–only use).

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
conda create -n monkey-gaze-tracker -c conda-forge python=3.8 pytables hdf5 lzo opencv numpy pandas matplotlib scikit-learn scikit-image scipy tqdm statsmodels
conda activate monkey-gaze-tracker
```

#### Alternatively, Using pip with a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

Install the dependencies (including DeepLabCut):

```bash
pip install -r requirements.txt
```

### 4. Prepare Your Data

- **Video Files:**  
  Place your trial video files (e.g., `1.mp4`, `2.mp4`, etc.) into a folder (for example, `videos/input/`).

- **Touch Event Files:**  
  For each video, create a corresponding touch event file (e.g., `1.txt` or `1.csv`) in CSV format with a header:
  ```
  timestamp,x,y
  ```
  Sample content:
  ```
  2025-02-24T18:41:57.2864969-05:00,16,15
  2025-02-24T18:41:58.6795674-05:00,34,25
  ```

### 5. Train Your DLC Model

1. **Launch the DLC GUI:**  
   After installing, start the DLC GUI with:
   ```bash
   python -m deeplabcut
   ```
2. **Create a New Project:**  
   In the GUI, create a new project by specifying your project name, your name, the video(s) to be analyzed, and the list of body parts (for example, define one keypoint such as `eye` or `pupil_center`).
3. **Label Frames:**  
   Use the GUI to label the eye landmark on a diverse set of frames.
4. **Train the Network:**  
   Use the **“Train Network”** option in the GUI.  
   *Tip:* For CPU–only systems, edit your project’s `config.yaml` and change the network type from `resnet_50` to a lighter option (e.g., `mobilenet_v2_1.0`) for faster training/inference.
5. **Evaluate the Network:**  
   Once training is complete, evaluate the network to ensure detection accuracy.

### 6. Update the Eye Detection Function

Update the eye detection function in `src/detect_eye.py` so that it:
- Loads your trained DLC model (via your project config).
- Processes each video frame to detect the eye landmark.
- Returns the (x, y) coordinate for that landmark.

### 7. Calibration

Run the calibration script to compute the homography (calibration) matrix. This script uses the first four touch events from each touch event file to compute the mapping from raw eye coordinates to screen coordinates.

```bash
python src/calibrate.py
```

Calibration files (e.g., `calibration_matrix_1.npy` and `calibration_1.yaml`) will be saved in the `data/trained_model/` folder.

### 8. Analyze Videos

Once calibration is complete and your DLC-based eye detection is working, run the analysis script. This script:
- Loads each video (and corresponding touch event file, if available).
- Uses the updated eye detection function (with DLC) to extract eye coordinates.
- Applies the calibration matrix to map these coordinates to screen coordinates.
- Divides the screen into predefined regions (e.g., 110 regions) and computes the total time spent fixating in each region.
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

To generate plots or heatmaps of the gaze points, run the visualization script:

```bash
python src/visualize.py --csv path/to/your_gaze_data.csv
```

### Final Verification

After installation and setup, run the following command to verify that everything is correctly installed:

```bash
python -c "import tables, deeplabcut; print('Installation successful')"
```

You should see:

```
Installation successful
```

Happy tracking!
