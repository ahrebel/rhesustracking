# rhesusDLC

# Monkey Gaze Tracker

This project provides a Python-based eye-tracking software to analyze recorded videos of monkeys interacting with a touchscreen. It uses DeepLabCut (DLC) to detect eye landmarks and maps gaze to one of 110 screen sections, computing the total time spent on each section.

## Features
- Offline video processing (post-recording)
- DeepLabCut-based eye detection (with a placeholder for inference)
- Calibration and automatic recalibration (via a simple perspective transform)
- Screen division into 110 sections (configurable via YAML)
- Cross-platform support (Mac and Windows)
- Structured folder organization and easy setup with provided scripts



## Setup Instructions

1. **Clone the repository.**

2. **Create your Python environment:**
   - With Conda (recommended):  
     ```bash
     conda env create -f environment.yml
     conda activate monkey-gaze-tracker
     ```
   - Or with pip:  
     ```bash
     python -m venv venv
     source venv/bin/activate  # (or venv\Scripts\activate on Windows)
     pip install -r requirements.txt
     ```

3. **Alternatively, run the provided setup script:**
   - On Mac/Linux:  
     ```bash
     bash setup.sh
     ```
   - On Windows:  
     ```batch
     setup.bat
     ```

4. **Place your trial videos in the `videos/input/` folder.**

5. **Run calibration (to compute and save a calibration matrix):**
   ```bash
   python src/calibrate.py

6. 
```bash
python src/train_model.py
```

7. 
```bash
python src/analyze_video.py
```
8. (Optional) Show heatmap on screen
```bash
python src/visualize.py
```

