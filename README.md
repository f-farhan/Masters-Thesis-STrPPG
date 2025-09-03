
# `sTrPPG Code` Documentation

This is the code is used for rPPG signal extraction pipeline.  
It integrates multiple modules including video stabilization, ROI selection, Filtering, signal extraction, and heatmap generation.

---

## ⚙️ Description

- **Source Folder Setup**: Specify the directories containing source videos and output folders.
- **Processing Parameters**: Choose frame rate, filtering settings, heatmap overlay options.
- **Filtering Options**: Set Butterworth and bilateral filter parameters.
- **Stabilization**: Automatically extracts the most stable section and stabilizes it before analysis.
- **Output**: Generates various heatmaps, reference signals, and evaluation plots.

---

## Key Imported Modules

- `auto_stabilize_return_video` — Selects and stabilizes low-motion segments.
- `getROI_Image`, `getROI_Image_GPU` — ROI and reference ROI extraction.
- `getrPPG`, `getrPPGmean`, `getHRmean` — Signal estimation.
- `getPLOT`, `getSNR`, `getHR`, `getPowerMAP`, etc. — Analysis and heatmap generation.
- `overlay_heatmap_with_manual_roi` — Overlay result visualization.
- `config.cf` — Contains configuration flags like `SELECT_ROI`, `DEBUG`, `METHOD`, etc.

---

## Processing Steps

### 1. Initialization
- Starts timer
- Defines video input and output paths

### 2. Video Stabilization
- Uses `auto_stabilize_return_video` to select and stabilize a the most stable section (10 seconds)
- Saves stabilized video for comparison of magnitudes

### 3. Parameters
- Set the frame rate according to the source
- Set Butterworth filter parameters
- Set bilateral filter parameters

### 4. Ground Truth File Handling
- Sets the name of the ground truth file
- Initializes empty lists for ground truth heart rate values, PPG signal trace, timestamps for the ground truth data.
- Combines the directory of the input video with the filename to form the full path to the ground truth file.
- Attempts to Load Ground Truth Data from input directory.

### 5. Directories and Configuration file
- Creates folders:
	- npy_files – for storing NumPy intermediate files.
	- heatmaps – for storing heatmap visualizations.


### 6. ROI & Reference ROI Selection
- Opens a GUI window (cv2.selectROI) for the user to manually select a rectangular ROI from the first video frame.
- Extracted ROI is used for signal processing
- Records bilateral filter parameters to Configuration.txt.

### 7. Reference Signal Extraction
- Uses `getrPPGmean()` to obtain mean signal from reference ROI
- Computes FFT spectrum of the signal

### 8. Full ROI rPPG Estimation
- Uses `getrPPG()` for pixel-level rPPG estimation
- Normalizes signals and logs method/filtering settings

### 6. Signal Quality Index (SQI)
- Computes template correlation SQI for a central pixel
- Plots and saves raw vs. normalized signal

### 7. Heatmap Computation
- If `cf.HEATMAP == 1`, generates:
  - MAE, Correlation, Phase, Amplitude, and SNR maps
  - Uses multiple methods like Hilbert, Lock-in, Neurokit, Mean, FindPeaks

### 8. Final Output & Stats
- Displays mean BPM from ground truth, sensor, and rPPG
- Displays SQI and SNR values
- Optionally overlays SNR map on hand image

---

## 🛠 Configuration File (`config.py`)

Parameters to configure:
- `cf.SELECT_ROI`: Use interactive ROI selection
- `cf.METHOD`: Signal extraction method (Green, POS, CHROM, G-R)
- `cf.FORMAT`: Input format (uint8, uint16)
- `cf.DEBUG`: Save intermediate outputs
- `cf.AFFICHAGE`: Show comparison plots

---

## ✅ Example (Typical Usage)

```
python main.py
```

To compare the average motion magnitudes between videos run:

```
python motion_magnitude.py
```

Make sure all dependencies are installed, the config is set, and paths to input video, output folder, and ground truth file are valid.

---

## 🗂 Output Files

- `ROI.npy`, `ROI_REF.npy`: Selected ROIs
- `rppg_ref.npy`, `timeTrace.npy`: Reference signal
- `pulseTraces.npy`: Full rPPG signals
- `*.png`: Heatmaps and FFT/signal plots
- `Configuration.txt`: Processing log
