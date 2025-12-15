# SAIM Analysis Engine v9.2

**Scientific Release: Bilateral Fusion & Adaptive Reliability Gating**

## Overview
This repository contains the source code for the **Spinal Active Inference Model (SAIM) Analysis Engine v9.2**. This computational framework is designed to quantify the transition from "Central Prediction Error Neglect" (Functional Spinal Fixation) to "Active Inference" (Plasticity).

## Key Features (v9.2 Final)
* **Bilateral Hemodynamic Tracking:** Utilizes **Modified Beer-Lambert Law (MBLL)** on both Left/Right Inner sensors (Ch13/7, Ch14/8) to estimate prefrontal Oxyhemoglobin ($\Delta [HbO_2]$).
* **Adaptive Reliability Gating (ARG):** Dynamically weights signal contributions based on real-time signal-to-noise ratio.
* **NCI Volatility (Criticality):** Computes the rolling volatility of the Neural Complexity Index (NCI) to detect phase transitions (Self-Organized Criticality).

## Hardware Compatibility
* **Device:** Interaxon Muse S (Gen 2 / Athena)
* **Sensor Configuration:**
    * **Left Inner:** Red (Ch13) / IR (Ch7)
    * **Right Inner:** Red (Ch14) / IR (Ch8)
    * **Outer Sensors:** Excluded to minimize hair-induced artifacts.

## File Structure
* `SAIM_Engine_v9_2.py`: The core analysis pipeline (Full Python Source).
* `SAIM_code_supplymentary1_v9_2.pdf`: Theoretical Methodology (S1).
* `SAIM_code_supplymentary2_v9_2.pdf`: Source Code Documentation (S2).

## Usage
1.  Place raw CSV files from Mind Monitor in the data directory.
2.  Configure `TARGET_ID` and `GROUP_TYPE` in the script.
3.  Run the analysis:
    ```bash
    python SAIM_Engine_v9_2.py
    ```

## Citation & Theory
For the mathematical derivation of NCI and NCI Volatility, please refer to **Supplementary Material S1**.

---
*© 2025 TIC-DO Institute of Vertebral Subluxation Research*
