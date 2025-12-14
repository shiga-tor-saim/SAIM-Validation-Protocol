import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Import Core Logic (Invariant)
from core_metrics import calc_metrics, W_DEFAULTS, EPS

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================================
#   Protocol Settings (Version 3.4)
# ==========================================================
class SAIMProtocol:
    # --- Analysis Parameters ---
    WINDOW_SEC = 10
    STEP_SEC = 2
    MIN_DATA_POINTS = 5

    # --- UNIVERSAL PHASE ORDER (v3.4 Extended) ---
    PHASE_ORDER = [
        '01_Pre', '02_PostImmed', '03_PostStress', '04_PostRest',
        '05_RescueImmed', '06_RescuePost'
    ]

    # --- Dynamic Labels ---
    LABELS_REAL = {
        '01_Pre': 'I: Pre',
        '02_PostImmed': 'II: PostImmed',
        '03_PostStress': 'III: PostStress',
        '04_PostRest': 'IV: PostRest'
    }

    LABELS_SHAM = {
        '01_Pre': 'I: Pre',
        '02_PostImmed': 'II: PostImmed',
        '03_PostStress': 'III: PostStress',
        '04_PostRest': 'IV: PostRest',
        '05_RescueImmed': 'V: RescueImmed',
        '06_RescuePost': 'VI: RescuePost'
    }

# ==========================================================
#   Main Processor Class
# ==========================================================
class SAIMAnalyzer:
    def __init__(self, subject_id, file_map, is_sham=False):
        self.subject_id = subject_id
        self.file_map = file_map
        self.is_sham = is_sham
        self.df_final = pd.DataFrame()
        self.active_metrics = set()

        self.labels = SAIMProtocol.LABELS_SHAM if self.is_sham else SAIMProtocol.LABELS_REAL

    def _load_clean_data(self, filepath):
        try:
            if not os.path.exists(filepath):
                # Try checking in current directory if full path fails
                if os.path.exists(os.path.basename(filepath)):
                    filepath = os.path.basename(filepath)
                else:
                    print(f"[Error] File not found: {filepath}")
                    return None

            df = pd.read_csv(filepath, low_memory=False)
            if 'TimeStamp' not in df.columns: return None
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
            df = df.dropna(subset=['TimeStamp']).sort_values('TimeStamp')
            if 'Elements' in df.columns: df = df[df['Elements'].isna()]

            keywords = ['Gamma', 'Delta', 'Alpha', 'Accelerometer', 'Heart', 'Optics', 'Optical', 'AUX', 'Infrared']
            cols = [c for c in df.columns if any(k in c for k in keywords)]
            for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df
        except: return None

    def run_analysis(self):
        mode_label = "SHAM (6-Phase)" if self.is_sham else "REAL (4-Phase)"
        print(f"--- Processing {self.subject_id} [{mode_label}] ---")
        c = SAIMProtocol
        raw_data = []

        for key in c.PHASE_ORDER:
            if key not in self.file_map: continue
            df = self._load_clean_data(self.file_map[key])
            if df is None: continue

            duration = (df['TimeStamp'].iloc[-1] - df['TimeStamp'].iloc[0]).total_seconds()
            fs_est = len(df) / duration if duration > 0 else 1.0
            label = self.labels.get(key, key)
            print(f" > Processing {key} ({label}): {fs_est:.1f}Hz")

            curr, end = df['TimeStamp'].iloc[0], df['TimeStamp'].iloc[-1]
            while curr < end:
                nxt = curr + pd.Timedelta(seconds=c.WINDOW_SEC)
                df_win = df[(df['TimeStamp'] >= curr) & (df['TimeStamp'] < nxt)]
                if len(df_win) >= 3:
                    # CALL CORE METRICS (From core_metrics.py)
                    metrics = calc_metrics(df_win, fs_est)
                    raw_data.append((label, key, *metrics))
                curr += pd.Timedelta(seconds=c.STEP_SEC)

        if not raw_data:
            print("[Error] No valid data extracted.")
            return

        df_raw = pd.DataFrame(raw_data, columns=['Phase', 'PhaseKey', 'FSI', 'SOM', 'PE', 'AUT', 'HEMO', 'RR'])

        # Adaptive Weighting: Explicitly excludes RR from NCI integration
        # Uses W_DEFAULTS imported from core_metrics
        valid_weights = {m: W_DEFAULTS[m] for m in ['FSI', 'SOM', 'AUT', 'HEMO']
                         if df_raw[m].count() > 0 and df_raw[m].std() > 1e-6}
        self.active_metrics = set(valid_weights.keys())
        total_w = sum(valid_weights.values())
        if total_w > 0:
            for k in valid_weights: valid_weights[k] /= total_w

        print(f" > Active Weights: {valid_weights}")

        results = []
        for _, row in df_raw.iterrows():
            score = sum(row[m] * w for m, w in valid_weights.items() if not np.isnan(row[m]))
            if not np.isnan(row['PE']): score -= row['PE'] * W_DEFAULTS['PE']

            nci = 1.0 / (1.0 + np.exp(-score))
            active_vals = [row[m] for m in valid_weights.keys() if not np.isnan(row[m])]
            f_val = (1.0 - np.mean(active_vals)) + (row['PE'] * 0.5) if active_vals else np.nan

            results.append({
                'Phase': row['Phase'], 'PhaseKey': row['PhaseKey'],
                'NCI_Z': nci, 'F_Z': f_val, 'PE_Z': row['PE'], 'HEMO_Z': row['HEMO']
            })

        self.df_final = pd.DataFrame(results)
        self._normalize_and_plot()

    def _normalize_and_plot(self):
        base_df = self.df_final[self.df_final['PhaseKey'] == '01_Pre']
        if base_df.empty: base_df = self.df_final

        # Standardization (Raw -> Z-Score)
        for m in ['NCI_Z', 'PE_Z', 'F_Z', 'HEMO_Z']:
            mean, std = base_df[m].mean(), base_df[m].std()
            if np.isnan(std) or std < 1e-9: std = 1.0
            self.df_final[m] = (self.df_final[m] - mean) / std # Overwrite with Z-score

            # Trends & Volatility (Bandwidth)
            self.df_final[f'{m}_Trend'] = self.df_final[m].interpolate().rolling(5, center=True).mean()
            self.df_final[f'{m}_Std'] = self.df_final[m].interpolate().rolling(5, center=True).std()

        # --- Plotting ---
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(14, 18))
        plt.suptitle(f"SAIM Analysis: {self.subject_id} (v3.4)", fontsize=16, fontweight='bold')

        # 1. NCI & PE (Neuro-Somatic)
        ax1 = plt.subplot(3, 1, 1)
        df_p = self.df_final.reset_index(drop=True)
        ax1.plot(df_p.index, df_p['NCI_Z_Trend'], color='#00BFFF', label='NCI', lw=2.5)
        ax1.fill_between(df_p.index, df_p['NCI_Z_Trend']-df_p['NCI_Z_Std'], df_p['NCI_Z_Trend']+df_p['NCI_Z_Std'], color='#00BFFF', alpha=0.15)
        ax1.plot(df_p.index, df_p['PE_Z_Trend'], color='#FF4500', label='PE', lw=2.5)
        self._add_boundaries(ax1, df_p)
        ax1.set_title(f"1. Neuro-Somatic Dynamics (Active: {list(self.active_metrics)})", fontsize=14)
        ax1.legend(loc='upper left')

        # 2. Metabolic
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(df_p.index, df_p['F_Z_Trend'], color='magenta', label='F', lw=2.5)
        if 'HEMO' in self.active_metrics:
            ax2.plot(df_p.index, df_p['HEMO_Z_Trend'], color='red', label='HEMO', lw=2.5, linestyle='--')
        self._add_boundaries(ax2, df_p)
        ax2.set_title("2. Metabolic Cost & Flow", fontsize=14)
        ax2.legend(loc='upper left')

        # 3. Stats
        ax3 = plt.subplot(3, 1, 3)
        df_melt = self.df_final.melt(id_vars='Phase', value_vars=['NCI_Z', 'PE_Z', 'F_Z', 'HEMO_Z'], var_name='Metric', value_name='Z-Score')
        sns.boxplot(data=df_melt, x='Phase', y='Z-Score', hue='Metric', palette='viridis', ax=ax3, showfliers=False)
        ax3.axhline(0, color='black')
        ax3.set_title("3. Statistical Distribution", fontsize=14)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"SAIM_Result_{self.subject_id}_v3.4.png")
        print(f"Graph saved: SAIM_Result_{self.subject_id}_v3.4.png")

        # --- OUTPUT: Requested Format ---
        print("\n=== Numerical Summary (Mean Z-Score) ===")
        print(self.df_final.groupby('Phase', sort=False)[['NCI_Z', 'PE_Z', 'F_Z', 'HEMO_Z']].mean())

        print("\n=== NCI Bandwidth (Volatility Analysis) ===")
        # Calculate mean of rolling standard deviation for NCI
        print(self.df_final.groupby('Phase', sort=False)['NCI_Z_Std'].mean())

        # Save CSVs
        self.df_final.to_csv(f"SAIM_Data_{self.subject_id}_v3.4.csv", index=False)

    def _add_boundaries(self, ax, df):
        boundaries = df.groupby('PhaseKey').apply(lambda x: x.index[-1]).tolist()[:-1]
        for b in boundaries: ax.axvline(b, color='gray', linestyle=':', alpha=0.8)

# ==========================================================
#   UNIVERSAL LAUNCHER
# ==========================================================
def auto_run_subject():
    # ------------------------------------------------------
    # [USER SETTINGS] - Configure here
    # ------------------------------------------------------
    TARGET_ID = 'S99'       # Example: 'S00', 'S99'
    GROUP_TYPE = 'Real'     # 'Real' or 'Sham'
    DATA_FOLDER = 'Data'    # Folder containing CSVs
    # ------------------------------------------------------

    # Auto File Map
    is_sham = (GROUP_TYPE == 'Sham')
    files = {}
    file_defs = {
        '01_Pre':         f'{TARGET_ID}_01_Pre.csv',
        '02_PostImmed':   f'{TARGET_ID}_02_PostImmed.csv',
        '03_PostStress':  f'{TARGET_ID}_03_PostStress.csv',
        '04_PostRest':    f'{TARGET_ID}_04_PostRest.csv',
    }
    if is_sham:
        file_defs.update({
            '05_RescueImmed': f'{TARGET_ID}_05_RescueImmed.csv',
            '06_RescuePost':  f'{TARGET_ID}_06_RescuePost.csv'
        })

    # Try to locate files flexibly
    for key, fname in file_defs.items():
        files[key] = os.path.join(DATA_FOLDER, fname)

    # Execution
    subject_label = f"{TARGET_ID}_{GROUP_TYPE}"
    print(f"\n>>> Starting Analysis for: {subject_label}")
    analyzer = SAIMAnalyzer(subject_id=subject_label, file_map=files, is_sham=is_sham)
    analyzer.run_analysis()

if __name__ == "__main__":

    auto_run_subject()

