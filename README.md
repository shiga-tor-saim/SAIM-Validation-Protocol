# SAIM Analysis Engine v9.2

**Scientific Release: Bilateral Fusion & Adaptive Reliability Gating**

## Overview
This repository contains the source code for the **Spinal Active Inference Model (SAIM) Analysis Engine v9.2**. This computational framework is designed to quantify the transition from "Central Prediction Error Neglect" (Functional Spinal Fixation) to "Active Inference" (Plasticity) following Specific Input Perturbation (SIP).

## Key Features (v9.2)
* **Bilateral Hemodynamic Tracking:** Utilizes **Modified Beer-Lambert Law (MBLL)** on both Left and Right Inner optical sensors to estimate prefrontal Oxyhemoglobin ($\Delta [HbO_2]$) volatility.
* **Adaptive Reliability Gating (ARG):** Dynamically weights signal contributions based on real-time signal-to-noise ratio, ensuring robustness against sensor dropout.
* **Neuro-Somatic Integration:** Computes the **Neural Complexity Index (NCI)** by integrating Hemodynamics (HEMO), Neural Precision (FSI), Somatic Order (SOM), and Autonomic Complexity (AUT).

## Hardware Compatibility
* **Device:** Interaxon Muse S (Gen 2 / Athena)
* **Sensor Configuration:**
    * **Left Inner:** Red (Ch13) / IR (Ch7)
    * **Right Inner:** Red (Ch14) / IR (Ch8)
    * **Outer Sensors:** Excluded to minimize hair-induced artifacts.

## File Structure
* `SAIM_Engine_v9_2.py`: The core analysis pipeline (matches Supplementary Material S2).
* `SAIM_code_supplymentary1_v9_2.pdf`: Theoretical Methodology (S1).
* `SAIM_code_supplymentary2_v9_2.pdf`: Source Code Documentation (S2).

## Usage
1.  Place raw CSV files from Mind Monitor in the data directory.
2.  Configure `TARGET_ID` and `GROUP_TYPE` in the launcher script.
3.  Run the analysis:
    ```bash
    python SAIM_Engine_v9_2.py
    ```

## Citation & Theory
For the mathematical derivation of the HEMO index and NCI integration logic, please refer to **Supplementary Material S1** included in this repository.

---
*© 2025 TIC-DO Institute of Vertebral Subluxation Research*
