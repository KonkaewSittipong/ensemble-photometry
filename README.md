# Ensemble Photometry Pipeline

A Python-based pipeline designed for precision photometry and atmospheric correction using the Ensemble Calibration method. This tool is optimized for high-cadence astronomical data (HiPERCAM/hcam) and research involving binary pulsar systems.

## Key Features
- **Automated Experiment Tracking:** Automatically generates incremented run directories (run001, run002, etc.) to prevent overwriting previous results.
- **Iterative Star Rejection:** Automatically identifies and removes variable or low-quality reference stars based on global RMS targets.
- **Diagnostic Plotting:** Automatically saves diagnostic plots, including sky background, contrast ratios, and variability residuals, for every iteration.
- **Time and Extinction Correction:** Integrated barycentric time conversion (MJD to BJD) and airmass-dependent extinction correction using astropy.

## Prerequisites
The following Python packages are required:
- numpy
- pandas
- matplotlib
- astropy

## Usage
1. Prepare your input log files (e.g., hcam logs).
2. Configure paths and parameters within run_Ensemble.py.
3. Run the pipeline via the terminal:
   ```bash
   python run_Ensemble.py
