import numpy as np
from scipy.signal import detrend, butter, filtfilt
from scipy.stats import entropy

# ==========================================================
#   CORE LOGIC (Mathematically Consistent)
# ==========================================================

# --- Constants ---
EPS = 1e-9

# --- Adaptive Weights ---
# Note: RR is calculated but excluded from integration defaults
W_DEFAULTS = {
    'FSI': 0.35, 'SOM': 0.35, 'AUT': 0.15, 'HEMO': 0.15, 'PE': 0.30
}

def estimate_hemo_rr(df_win, fs_est):
    """
    HEMO (Metabolic Flow) and RR (Respiration Rate) estimation.
    Algorithm: Log-Sigmoid for HEMO, Bandpass+FFT for RR.
    """
    ppg_candidates = [c for c in df_win.columns if 'Optics' in c or 'Optical' in c or 'Infrared' in c or 'AUX' in c]
    hemo, rr = np.nan, np.nan
    if ppg_candidates:
        valid_cols = [c for c in ppg_candidates if df_win[c].notna().sum() > 3]
        if valid_cols:
            sig = df_win[valid_cols].mean(axis=1).dropna().values
            if len(sig) > 5:
                # HEMO: Metabolic Flow (Log-Sigmoid)
                std_val = np.std(sig)
                hemo = 1.0 / (1.0 + np.exp(-(np.log(std_val) - 2.0))) if std_val > 0 else 0.0

                # RR: Respiration Rate (Bandpass Filter + FFT)
                if fs_est >= 2.0:
                    try:
                        b, a = butter(2, [0.1/(fs_est/2), 0.5/(fs_est/2)], btype='band')
                        filtered = filtfilt(b, a, sig)
                        freqs = np.fft.rfftfreq(len(filtered), 1/fs_est)
                        fft_mag = np.abs(np.fft.rfft(filtered))
                        mask = (freqs > 0.1) & (freqs < 0.5)
                        if np.any(mask):
                            rr = freqs[mask][np.argmax(fft_mag[mask])] * 60
                    except: pass
    return hemo, rr

def calc_metrics(df_win, fs_est):
    """
    Calculates FSI, SOM, PE, AUT, HEMO, RR based on established mathematical models.
    """
    # 1. Neural Precision (FSI)
    g_cols = [col for col in df_win.columns if 'Gamma' in col]
    d_cols = [col for col in df_win.columns if 'Delta' in col]
    if g_cols and d_cols:
        g_val = np.nanmedian(np.nanmean(df_win[g_cols], axis=1))
        d_val = np.nanmedian(np.nanmean(df_win[d_cols], axis=1))
        # Log-Ratio with stabilization
        fsi_raw = np.log((g_val + EPS) / d_val) if min(df_win[g_cols].min().min(), df_win[d_cols].min().min()) >= 0 else g_val - d_val
        fsi = 1 / (1 + np.exp(-fsi_raw))
    else: fsi = np.nan

    # 2. Somatic Order (SOM)
    acc_cols = [col for col in df_win.columns if 'Accelerometer' in col]
    som = 1.0 / (1.0 + np.std(np.sqrt(np.sum(df_win[acc_cols]**2, axis=1)))) if acc_cols else np.nan

    # 3. Prediction Error (PE)
    a_cols = [col for col in df_win.columns if 'Alpha' in col]
    pe = np.nan
    if a_cols:
        a_val = np.nanmean(df_win[a_cols], axis=1)
        pe = np.std(detrend(a_val)) / (np.mean(np.abs(a_val)) + EPS)

    # 4. Autonomic Complexity (AUT)
    aut = np.nan
    if 'Heart_Rate' in df_win.columns:
        hr = df_win['Heart_Rate'].dropna().values
        if len(hr) > 3:
            counts, _ = np.histogram(hr, bins='fd')
            aut = entropy(counts/np.sum(counts)) / (np.log(len(counts)+1) + EPS)

    # 5. Hemodynamic Capacity (HEMO) & 6. RR (Validation)
    hemo, rr = estimate_hemo_rr(df_win, fs_est)

    return fsi, som, pe, aut, hemo, rr