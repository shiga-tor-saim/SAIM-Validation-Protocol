"""
SAIM Engine v9.3: Spinal Active Inference Model Analysis Pipeline
Official Implementation for Manuscript Submission
Author: Takafumi Shiga
License: MIT
Dependencies: pandas, numpy, matplotlib, seaborn, scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from matplotlib.collections import LineCollection
from scipy.signal import detrend
import warnings
import os
import datetime
import platform
import random
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================================
# 1. Configuration & Physics Constants
# ==========================================================
class SAIMConfig:
    # --- Time-Series Analysis Parameters ---
    WINDOW_SEC = 10
    STEP_SEC = 2
    MIN_DATA_POINTS = 5
    EPS = 1e-9

    # --- fNIRS / MBLL Parameters ---
    # Extinction coefficients (mM^-1 cm^-1) for HbO/HbR
    # Matrix Row 1: 660nm (Red), Row 2: 850nm (IR)
    E = np.array([
        [0.087, 0.730], 
        [0.052, 0.032]
    ])
    DPF = 6.0  # Differential Pathlength Factor for adult forehead

    # --- Phase Definitions ---
    PHASE_ORDER = [
        '01_Pre', '02_PostImmed', '03_PostStress', '04_PostRest',
        '05_RescueImmed', '06_RescuePost'
    ]
    LABELS_DISPLAY = {
        '01_Pre': 'I: Pre', '02_PostImmed': 'II: PostImmed',
        '03_PostStress': 'III: PostStress', '04_PostRest': 'IV: PostRest',
        '05_RescueImmed': 'V: RescueImmed', '06_RescuePost': 'VI: RescuePost'
    }

# ==========================================================
# 2. Helper Functions: Robust Statistics
# ==========================================================
def robust_scale(series):
    """
    Robust Z-score normalization using Median and MAD.
    Formula: 0.6745 * (x - median) / MAD
    """
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad < 1e-9: return series - median
    return 0.6745 * (series - median) / mad

def bootstrap_ci_effect_size(data_pre, data_post, n_boot=2000):
    """
    Calculates Mean Difference and 95% Confidence Interval via Bootstrapping.
    Returns: (Mean_Diff, CI_Lower, CI_Upper)
    """
    if len(data_pre) < 3 or len(data_post) < 3:
        return np.nan, np.nan, np.nan
    diffs = []
    for _ in range(n_boot):
        res_pre = np.random.choice(data_pre, size=len(data_pre), replace=True)
        res_post = np.random.choice(data_post, size=len(data_post), replace=True)
        diffs.append(np.mean(res_post) - np.mean(res_pre))
    return np.mean(data_post)-np.mean(data_pre), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

# ==========================================================
# 3. Main Analyzer Class
# ==========================================================
class SAIMAnalyzer:
    def __init__(self, subject_id, file_map, is_sham=False):
        self.subject_id = subject_id
        self.file_map = file_map
        self.is_sham = is_sham
        self.df_final = pd.DataFrame()
        self.labels = SAIMConfig.LABELS_DISPLAY
        self.blind_id = f"Blind_{random.randint(1000, 9999)}"
        try: self.E_inv = np.linalg.inv(SAIMConfig.E)
        except: self.E_inv = None

    def _load_clean_data(self, filepath):
        try:
            if not os.path.exists(filepath): return None
            df = pd.read_csv(filepath, low_memory=False)
            if 'TimeStamp' not in df.columns: return None
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
            df = df.dropna(subset=['TimeStamp']).sort_values('TimeStamp').reset_index(drop=True)
            if 'Elements' in df.columns: df = df[df['Elements'].isna() | (df['Elements'] == '')]
            # Convert numeric columns
            cols_to_check = ['Optics', 'Gamma', 'Delta', 'Alpha', 'Accelerometer', 'Heart']
            cols_to_num = [c for c in df.columns if any(k in c for k in cols_to_check)]
            for c in cols_to_num: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df
        except: return None

    def _calculate_brain_hemo(self, df_win):
        c = SAIMConfig
        # Muse S Sensor Pairs for Left/Right Inner (Red/IR)
        sensor_pairs = [('Optics13', 'Optics7'), ('Optics14', 'Optics8')]
        valid_hbo_stds = []
        for col_red, col_ir in sensor_pairs:
            if col_red not in df_win.columns or col_ir not in df_win.columns: continue
            try:
                raw_red = df_win[col_red].replace(0, np.nan).dropna()
                raw_ir  = df_win[col_ir].replace(0, np.nan).dropna()
                common_idx = raw_red.index.intersection(raw_ir.index)
                if len(common_idx) < c.MIN_DATA_POINTS: continue
                
                # Modified Beer-Lambert Law (MBLL)
                raw_red, raw_ir = raw_red.loc[common_idx], raw_ir.loc[common_idx]
                mean_red, mean_ir = raw_red.mean(), raw_ir.mean()
                if mean_red <= 0 or mean_ir <= 0: continue
                
                od_red = -np.log(raw_red / mean_red)
                od_ir = -np.log(raw_ir / mean_ir)
                
                conc_matrix = self.E_inv @ np.vstack((od_red.values/c.DPF, od_ir.values/c.DPF))
                valid_hbo_stds.append(np.std(conc_matrix[0, :])) # HbO Volatility
            except: continue
            
        if not valid_hbo_stds: return np.nan
        # Neurovascular Flexibility (Sigmoid Normalization of HbO Variance)
        hemo = 1.0 / (1.0 + np.exp(-(np.log(np.mean(valid_hbo_stds) * 1000 + c.EPS) - 2.0)))
        return hemo

    def _calc_metrics(self, df_win):
        c = SAIMConfig
        # 1. FSI (Frontal Stability Index - EEG)
        g_cols, d_cols = [x for x in df_win.columns if 'Gamma' in x], [x for x in df_win.columns if 'Delta' in x]
        fsi = np.nan
        if g_cols and d_cols:
            g_val, d_val = df_win[g_cols].mean().mean(), df_win[d_cols].mean().mean()
            fsi_raw = np.log((g_val+c.EPS)/(d_val+c.EPS)) if df_win[g_cols].min().min() >= 0 else (g_val - d_val)
            fsi = 1 / (1 + np.exp(-fsi_raw))
            
        # 2. SOM (Micro-Kinematic Stability)
        acc_cols = [x for x in df_win.columns if 'Accelerometer' in x]
        som = 1.0 / (1.0 + np.std(np.sqrt(np.sum(df_win[acc_cols]**2, axis=1)))) if acc_cols else np.nan
        
        # 3. PE (Prediction Error / Alpha Instability)
        a_cols = [x for x in df_win.columns if 'Alpha' in x]
        pe = np.nan
        if a_cols:
            a_clean = np.nanmean(df_win[a_cols], axis=1)
            pe = np.std(detrend(a_clean)) / (np.mean(np.abs(a_clean)) + c.EPS)
            
        # 4. AUT (Autonomic Flexibility / Entropy)
        aut = np.nan
        if 'Heart_Rate' in df_win.columns:
            hr = df_win['Heart_Rate'].dropna().values
            if len(hr) > 3:
                cnt, _ = np.histogram(hr, bins='fd')
                aut = entropy(cnt/np.sum(cnt)) / (np.log(len(cnt)+1) + c.EPS)
                
        # 5. HEMO (Neurovascular Flexibility)
        hemo = self._calculate_brain_hemo(df_win)
        
        return fsi, som, pe, aut, hemo

    def run_analysis(self):
        print(f"--- Processing {self.subject_id} (SAIM v9.3) ---")
        c = SAIMConfig
        
        # FAIL-SAFE: Check for Baseline Data
        if '01_Pre' not in self.file_map:
            print("!!! CRITICAL ERROR: '01_Pre' (Baseline) MISSING. Analysis Aborted. !!!")
            return 

        raw_data = []
        for key in c.PHASE_ORDER:
            if key not in self.file_map: continue
            df = self._load_clean_data(self.file_map[key])
            if df is None: continue
            label = self.labels.get(key, key)
            
            curr, end = df['TimeStamp'].iloc[0], df['TimeStamp'].iloc[-1]
            while curr < end:
                nxt = curr + pd.Timedelta(seconds=c.WINDOW_SEC)
                df_win = df[(df['TimeStamp'] >= curr) & (df['TimeStamp'] < nxt)]
                if len(df_win) >= c.MIN_DATA_POINTS:
                    metrics = self._calc_metrics(df_win)
                    raw_data.append((label, key, *metrics))
                curr += pd.Timedelta(seconds=c.STEP_SEC)

        if not raw_data: 
            print("No valid data points generated.")
            return
        
        df_raw = pd.DataFrame(raw_data, columns=['Phase', 'PhaseKey', 'FSI', 'SOM', 'PE', 'AUT', 'HEMO'])
        
        # Equal Weight Integration
        available_metrics = [m for m in ['FSI', 'SOM', 'AUT', 'HEMO'] 
                             if df_raw[m].count() > 0 and df_raw[m].std() > 1e-9]
        weight = 1.0 / len(available_metrics) if available_metrics else 0
        print(f" > Integrated Metrics (Equal Weight): {available_metrics}")
        
        results = []
        for _, row in df_raw.iterrows():
            # Integration
            integration_score = sum(row[m] * weight for m in available_metrics if not np.isnan(row[m]))
            # Prediction Error (Cost)
            pe_val = row['PE'] if not np.isnan(row['PE']) else 0
            # NCI Calculation
            nci = 1.0 / (1.0 + np.exp(-(integration_score - (pe_val * 0.5))))
            # Free Energy Proxy (F)
            av = [row[m] for m in available_metrics if not np.isnan(row[m])]
            f_val = (1.0 - np.mean(av)) + (pe_val * 0.5) if av else np.nan
            
            results.append({
                'Phase': row['Phase'], 'PhaseKey': row['PhaseKey'],
                'NCI': nci, 'PE': pe_val, 'HEMO': row['HEMO'], 'F': f_val,
                'FSI': row['FSI'], 'SOM': row['SOM'], 'AUT': row['AUT']
            })
        
        self.df_final = pd.DataFrame(results)
        
        # Calculate Volatility (NCI_Vol)
        self.df_final['NCI_Vol'] = self.df_final['NCI'].interpolate().rolling(5, center=True).std()

        # Exports and Plots
        self._export_full_data()
        self._summary_table()
        self._robust_normalize_and_plot()
        self._export_blind_statistics()
        plot_phase_plane_trajectory(self.df_final, self.subject_id)

    def _export_full_data(self):
        out_name = f"SAIM_FullData_{self.subject_id}_v9.3.csv"
        cols = ['Phase', 'PhaseKey', 'NCI', 'NCI_Vol', 'F', 'PE', 'HEMO', 'SOM', 'AUT', 'FSI']
        save_cols = [c for c in cols if c in self.df_final.columns]
        self.df_final[save_cols].to_csv(out_name, index=False)
        print(f"[Data] Saved Full Time-Series: {out_name}")

    def _summary_table(self):
        df = self.df_final
        disp = ['NCI', 'NCI_Vol', 'F', 'PE', 'HEMO', 'SOM', 'AUT', 'FSI']
        summary = df.groupby('Phase', sort=False)[[c for c in disp if c in df.columns]].mean().round(4)
        print("\n" + "-"*80)
        print(f" SAIM v9.3 Summary | Subject: {self.subject_id}")
        print("-"*80)
        print(summary)
        print("-"*80)
        summary.to_csv(f"SAIM_Summary_{self.subject_id}_v9.3.csv")

    def _robust_normalize_and_plot(self):
        df = self.df_final
        base_df = df[df['PhaseKey'] == '01_Pre']

        for m in ['NCI', 'PE', 'HEMO', 'F', 'FSI', 'SOM', 'AUT', 'NCI_Vol']:
            if m not in df.columns: continue
            median_val = base_df[m].median()
            mad_val = np.median(np.abs(base_df[m] - median_val))
            if mad_val < 1e-9: mad_val = 1.0
            
            df[f'{m}_Z'] = 0.6745 * (df[m] - median_val) / mad_val
            df[f'{m}_Z_Trend'] = df[f'{m}_Z'].interpolate().rolling(5, center=True).mean()

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(12, 18))
        
        # 1. Dynamics
        ax1 = plt.subplot(3, 1, 1)
        if 'NCI_Z_Trend' in df.columns:
            ax1.plot(df.index, df['NCI_Z_Trend'], color='#00BFFF', label='NCI', lw=2)
        if 'PE_Z_Trend' in df.columns:
            ax1.plot(df.index, df['PE_Z_Trend'], color='#FF4500', alpha=0.6, label='PE')
        self._add_boundaries(ax1, df); ax1.legend(loc='upper right'); ax1.set_title(f"1. Neuro-Somatic Dynamics ({self.subject_id})")

        # 2. Cost & Flex
        ax2 = plt.subplot(3, 1, 2)
        if 'HEMO_Z_Trend' in df.columns:
            ax2.plot(df.index, df['HEMO_Z_Trend'], color='red', linestyle='--', label='HEMO')
        if 'F_Z_Trend' in df.columns:
            ax2.plot(df.index, df['F_Z_Trend'], color='magenta', alpha=0.8, label='F (Free Energy)')
        self._add_boundaries(ax2, df); ax2.legend(loc='upper right'); ax2.set_title("2. Flexibility & Cost")
        
        # 3. Boxplots (Including F)
        ax3 = plt.subplot(3, 1, 3)
        plot_keys = [k for k in ['NCI_Z', 'PE_Z', 'HEMO_Z', 'F_Z'] if k in df.columns]
        if plot_keys:
            df_melt = df.melt(id_vars='Phase', value_vars=plot_keys, var_name='Metric', value_name='Robust Z-Score')
            sns.boxplot(data=df_melt, x='Phase', y='Robust Z-Score', hue='Metric', ax=ax3, showfliers=False)
        ax3.axhline(0, color='black', linestyle=':'); ax3.set_title("3. Distribution Analysis")

        plt.tight_layout()
        plt.savefig(f"SAIM_Result_{self.subject_id}_v9.3.png")
        print(f"[Graph] Saved: SAIM_Result_{self.subject_id}_v9.3.png")
        plt.show(); plt.close(fig)

    def _add_boundaries(self, ax, df):
        if 'PhaseKey' not in df.columns: return
        boundaries = df.reset_index().groupby('PhaseKey')['index'].max().tolist()[:-1]
        for b in boundaries: ax.axvline(b, color='gray', linestyle=':', alpha=0.7)

    def _export_blind_statistics(self):
        print("\n" + "="*60)
        print(f" SAIM v9.3 Statistical Report | Blind ID: {self.blind_id}")
        print("="*60)
        pre, post = self.df_final[self.df_final['PhaseKey'] == '01_Pre'], self.df_final[self.df_final['PhaseKey'] == '04_PostRest']
        stats_res = []
        if not pre.empty and not post.empty:
            for metric in ['NCI', 'PE', 'HEMO', 'F']:
                if metric not in pre.columns: continue
                m_diff, low, high = bootstrap_ci_effect_size(pre[metric].dropna().values, post[metric].dropna().values)
                sig = "*" if (low > 0 or high < 0) else "ns"
                print(f" {metric:10s} | Diff: {m_diff:.3f} [{low:.3f}, {high:.3f}] {sig}")
                stats_res.append({'Blind_ID': self.blind_id, 'Metric': metric, 'Mean_Diff': m_diff, 'CI_Lower': low, 'CI_Upper': high})
        if stats_res: pd.DataFrame(stats_res).to_csv(f"SAIM_Stats_{self.blind_id}_v9.3.csv", index=False)
        meta = {'Version': 'SAIM v9.3', 'Subject_ID': self.subject_id, 'Blind_ID': self.blind_id, 'Timestamp': datetime.datetime.now().isoformat()}
        pd.DataFrame([meta]).to_csv(f"META_{self.subject_id}_v9.3.csv", index=False)
        print("="*60 + "\n")

# ==========================================================
# 4. Trajectory Plot
# ==========================================================
def plot_phase_plane_trajectory(df, subject):
    if df.empty or 'PE_Z_Trend' not in df.columns or 'NCI_Z_Trend' not in df.columns: return
    x, y = df['PE_Z_Trend'].dropna(), df['NCI_Z_Trend'].dropna()
    common_idx = x.index.intersection(y.index)
    if len(common_idx) < 5: return
    x, y = x.loc[common_idx], y.loc[common_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='coolwarm', norm=plt.Normalize(np.linspace(0,1,len(x)).min(), np.linspace(0,1,len(x)).max()), alpha=0.7)
    lc.set_array(np.linspace(0, 1, len(x))); lc.set_linewidth(2.5); ax.add_collection(lc)
    ax.plot(x.iloc[0], y.iloc[0], 'o', color='blue', markersize=10, label='Start')
    ax.plot(x.iloc[-1], y.iloc[-1], 'X', color='red', markersize=12, label='End')
    
    for i, ph in enumerate(df['PhaseKey'].unique()):
        idx = df[df['PhaseKey'] == ph].index.intersection(common_idx)
        if len(idx) > 0:
            ax.text(x.loc[idx].mean(), y.loc[idx].mean(), str(i+1), fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='circle,pad=0.4'))

    ax.set_xlabel('Prediction Error (PE) [Robust Z-Score]')
    ax.set_ylabel('Neural Complexity (NCI) [Robust Z-Score]')
    ax.set_title(f'Phase Space Trajectory: {subject}\n(Hypothesis: Neuro-Somatic State Re-organization)')
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend()
    plt.tight_layout(); plt.savefig(f"Trajectory_{subject}_v9.3.png")
    print(f"[Trajectory] Saved: Trajectory_{subject}_v9.3.png")
    plt.show(); plt.close(fig)

# ==========================================================
# 5. Execution Block
# ==========================================================
def auto_run_subject():
    # --- Target Settings ---
    TARGET_ID='S'99; GROUP='Real'; DATA='/content/'
    # -----------------------
    files={}; phs={'01_Pre':'01_Pre','02_PostImmed':'02_PostImmed','03_PostStress':'03_PostStress','04_PostRest':'04_PostRest','05_RescueImmed':'05_RescueImmed','06_RescuePost':'06_RescuePost'}
    try: allf=os.listdir(DATA)
    except: return
    for k,s in phs.items():
        m=[f for f in allf if TARGET_ID in f and s in f]
        if m: files[k]=os.path.join(DATA,m[0])
    if not files: print(f"No files for {TARGET_ID}"); return
    a=SAIMAnalyzer(f"{TARGET_ID}_{GROUP}",files,is_sham=(GROUP=='Sham'))
    a.run_analysis()

if __name__=="__main__":
    auto_run_subject()
