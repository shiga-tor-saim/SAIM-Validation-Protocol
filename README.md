# SAIM Validation Analysis Framework (Protocol v3.4)

This repository contains the source code for the physiological data analysis used in the study **"Validation of the Spinal Active Inference Model (SAIM): A Neuro-Somatic Approach to Entropy Minimization"**.

The analysis framework is designed to process multi-modal physiological signals (EEG, Accelerometer, PPG) to compute the **Free Energy Index (FSI)**, **Neural Complexity Index (NCI)**, and **Hemodynamic Synchronization (HEMO)**.

## 📂 Repository Structure

To ensure scientific integrity and reproducibility, the codebase is strictly separated into **invariant mathematical logic** and **protocol-specific execution flow**:

* **`src/core_metrics.py`** (Invariant)
  * Contains the core mathematical algorithms for entropy calculation, band-pass filtering, and synchronization metrics.
  * **Note:** The mathematical definitions in this module have remained **consistent throughout Phase 1 and Phase 2b**, ensuring no p-hacking or algorithmic alteration occurred during the study.

* **`src/main_analysis.py`** (Updated to v3.4)
  * Handles data loading, windowing, and batch processing.
  * This module was updated to **Version 3.4** solely to accommodate the expanded experimental design (e.g., the 6-phase sequence in the Sham group including the Rescue phase), while calling the immutable functions from `core_metrics.py`.

## ⚙️ Requirements

* Python 3.8+
* Dependencies:
  * `numpy`
  * `pandas`
  * `scipy`
  * `matplotlib`
  * `seaborn`

## 🚀 Usage

1. Place your raw CSV data in a `Data` folder.
2. Run the main analysis script:

```bash
python src/main_analysis.py