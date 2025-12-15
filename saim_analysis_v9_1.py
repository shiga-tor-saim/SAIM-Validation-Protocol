import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import detrend
from scipy.stats import entropy
import warnings
import os

# 警告の抑制
warnings.filterwarnings('ignore')

# ==========================================================
#   SAIM Config v9.1 (Robust / Holy Grail Mapping + Output)
# ==========================================================
class SAIMConfig:
    # --- 解析パラメータ ---
    WINDOW_SEC = 10
    STEP_SEC = 2
    MIN_DATA_POINTS = 5
    EPS = 1e-9

    # --- MBLLパラメータ (Red 660nm / IR 850nm) ---
    E = np.array([
        [0.087, 0.730],  # 660nm -> [HbO, HbR]
        [0.052, 0.032]   # 850nm -> [HbO, HbR]
    ])
    DPF = 6.0 

    # --- 適応型重み付け (Adaptive Weights) ---
    W_DEFAULTS = {
        'FSI': 0.35, 'SOM': 0.35, 'AUT': 0.15, 'HEMO': 0.15, 'PE': 0.30
    }

    # --- フェーズ定義 ---
    PHASE_ORDER = [
        '01_Pre', '02_PostImmed', '03_PostStress', '04_PostRest',
        '05_RescueImmed', '06_RescuePost'
    ]
    
    LABELS_SHAM = {
        '01_Pre': 'I: Pre', '02_PostImmed': 'II: PostImmed',
        '03_PostStress': 'III: PostStress', '04_PostRest': 'IV: PostRest',
        '05_RescueImmed': 'V: RescueImmed', '06_RescuePost': 'VI: RescuePost'
    }
    LABELS_REAL = LABELS_SHAM 

# ==========================================================
#   Core Processing Class
# ==========================================================
class SAIMAnalyzer:
    def __init__(self, subject_id, file_map, is_sham=False):
        self.subject_id = subject_id
        self.file_map = file_map
        self.is_sham = is_sham
        self.df_final = pd.DataFrame()
        self.active_metrics = set()
        self.labels = SAIMConfig.LABELS_SHAM
        
        try:
            self.E_inv = np.linalg.inv(SAIMConfig.E)
        except:
            self.E_inv = None 

    def _load_clean_data(self, filepath):
        try:
            if not os.path.exists(filepath):
                if os.path.exists(os.path.basename(filepath)): filepath = os.path.basename(filepath)
                else: return None

            df = pd.read_csv(filepath, low_memory=False)
            if 'TimeStamp' not in df.columns: return None
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
            df = df.dropna(subset=['TimeStamp']).sort_values('TimeStamp').reset_index(drop=True)
            
            cols_to_check = ['Optics', 'Gamma', 'Delta', 'Alpha', 'Accelerometer', 'Heart']
            cols_to_num = [c for c in df.columns if any(k in c for k in cols_to_check)]
            for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
            
            return df
        except: return None

    def _calculate_brain_hemo(self, df_win):
        """Optics13/7 を使用してHEMOを計算"""
        c = SAIMConfig
        col_red = 'Optics13' # 660nm
        col_ir  = 'Optics7'  # 850nm
        
        if col_red not in df_win.columns or col_ir not in df_win.columns:
            return np.nan

        try:
            raw_red = df_win[col_red].replace(0, np.nan).dropna()
            raw_ir  = df_win[col_ir].replace(0, np.nan).dropna()
            
            common_idx = raw_red.index.intersection(raw_ir.index)
            if len(common_idx) < c.MIN_DATA_POINTS:
                return np.nan
                
            raw_red = raw_red.loc[common_idx]
            raw_ir = raw_ir.loc[common_idx]

            mean_red = raw_red.mean()
            mean_ir = raw_ir.mean()
            if mean_red <= 0 or mean_ir <= 0: return np.nan

            od_red = -np.log(raw_red / mean_red)
            od_ir  = -np.log(raw_ir / mean_ir)
            
            od_matrix = np.vstack((od_red.values / c.DPF, od_ir.values / c.DPF))
            conc_matrix = self.E_inv @ od_matrix
            hbo_series = conc_matrix[0, :]
            
            hbo_std = np.std(hbo_series)
            scale_factor = 1000 
            hemo = 1.0 / (1.0 + np.exp(-(np.log(hbo_std * scale_factor + c.EPS) - 2.0)))
            
            return hemo
        except Exception:
            return np.nan

    def _calc_metrics(self, df_win, fs_est):
        c = SAIMConfig
        g_cols = [col for col in df_win.columns if 'Gamma' in col]
        d_cols = [col for col in df_win.columns if 'Delta' in col]
        fsi = np.nan
        if g_cols and d_cols:
            g_val = df_win[g_cols].mean().mean()
            d_val = df_win[d_cols].mean().mean()
            is_log = (df_win[g_cols].min().min() < 0)
            try:
                fsi_raw = (g_val - d_val) if is_log else np.log((g_val+c.EPS)/(d_val+c.EPS))
                fsi = 1 / (1 + np.exp(-fsi_raw))
            except: pass

        acc_cols = [col for col in df_win.columns if 'Accelerometer' in col]
        som = np.nan
        if acc_cols:
            mag = np.sqrt(np.sum(df_win[acc_cols]**2, axis=1))
            som = 1.0 / (1.0 + np.std(mag))

        a_cols = [col for col in df_win.columns if 'Alpha' in col]
        pe = np.nan
        if a_cols:
            a_val = np.nanmean(df_win[a_cols], axis=1)
            mask = np.isfinite(a_val)
            a_clean = a_val[mask]
            if len(a_clean) > c.MIN_DATA_POINTS:
                try:
                    pe = np.std(detrend(a_clean)) / (np.mean(np.abs(a_clean)) + c.EPS)
                except: pe = np.nan

        aut = np.nan
        if 'Heart_Rate' in df_win.columns:
            hr = df_win['Heart_Rate'].dropna().values
            if len(hr) > 3:
                counts, _ = np.histogram(hr, bins='fd')
                aut = entropy(counts/np.sum(counts)) / (np.log(len(counts)+1) + c.EPS)

        hemo = self._calculate_brain_hemo(df_win)

        return fsi, som, pe, aut, hemo, np.nan

    def run_analysis(self):
        mode_label = "SHAM" if self.is_sham else "REAL"
        print(f"--- Processing {self.subject_id} [{mode_label}] ---")
        print(f"--- Engine: v9.1 (With Numerical Output) ---")
        
        c = SAIMConfig
        raw_data = []

        for key in c.PHASE_ORDER:
            if key not in self.file_map: continue
            df = self._load_clean_data(self.file_map[key])
            if df is None: continue

            label = self.labels.get(key, key)
            print(f" > Processing {key} ({label})...")

            curr, end = df['TimeStamp'].iloc[0], df['TimeStamp'].iloc[-1]
            while curr < end:
                nxt = curr + pd.Timedelta(seconds=c.WINDOW_SEC)
                df_win = df[(df['TimeStamp'] >= curr) & (df['TimeStamp'] < nxt)]
                if len(df_win) >= c.MIN_DATA_POINTS:
                    metrics = self._calc_metrics(df_win, 1.0)
                    raw_data.append((label, key, *metrics))
                curr += pd.Timedelta(seconds=c.STEP_SEC)

        if not raw_data:
            print("[Error] No valid data extracted.")
            return

        df_raw = pd.DataFrame(raw_data, columns=['Phase', 'PhaseKey', 'FSI', 'SOM', 'PE', 'AUT', 'HEMO', 'RR'])

        valid_weights = {m: c.W_DEFAULTS[m] for m in ['FSI', 'SOM', 'AUT', 'HEMO']
                         if df_raw[m].count() > 0 and df_raw[m].std() > 1e-6}
        
        total_w = sum(valid_weights.values())
        if total_w > 0:
            for k in valid_weights: valid_weights[k] /= total_w
        
        self.active_metrics = set(valid_weights.keys())
        print(f" > Active Metrics for NCI: {list(self.active_metrics)}")

        results = []
        for _, row in df_raw.iterrows():
            score = sum(row[m] * w for m, w in valid_weights.items() if not np.isnan(row[m]))
            if not np.isnan(row['PE']): score -= row['PE'] * c.W_DEFAULTS['PE']
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

        for m in ['NCI_Z', 'PE_Z', 'F_Z', 'HEMO_Z']:
            mean, std = base_df[m].mean(), base_df[m].std()
            if np.isnan(std) or std < 1e-9: std = 1.0
            self.df_final[m] = (self.df_final[m] - mean) / std 
            self.df_final[f'{m}_Trend'] = self.df_final[m].interpolate().rolling(5, center=True).mean()
            self.df_final[f'{m}_Std'] = self.df_final[m].interpolate().rolling(5, center=True).std()

        # Plotting
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(14, 18))
        plt.suptitle(f"SAIM Analysis: {self.subject_id} (v9.1)", fontsize=16, fontweight='bold')

        ax1 = plt.subplot(3, 1, 1)
        df_p = self.df_final.reset_index(drop=True)
        ax1.plot(df_p.index, df_p['NCI_Z_Trend'], color='#00BFFF', label='NCI', lw=2.5)
        ax1.fill_between(df_p.index, df_p['NCI_Z_Trend']-df_p['NCI_Z_Std'], df_p['NCI_Z_Trend']+df_p['NCI_Z_Std'], color='#00BFFF', alpha=0.15)
        ax1.plot(df_p.index, df_p['PE_Z_Trend'], color='#FF4500', label='PE', lw=2.5)
        self._add_boundaries(ax1, df_p)
        ax1.set_title(f"1. Neuro-Somatic Dynamics", fontsize=14)
        ax1.legend(loc='upper left')

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(df_p.index, df_p['F_Z_Trend'], color='magenta', label='F', lw=2.5)
        if 'HEMO' in self.active_metrics:
            ax2.plot(df_p.index, df_p['HEMO_Z_Trend'], color='red', label='HEMO', lw=2.5, linestyle='--')
        self._add_boundaries(ax2, df_p)
        ax2.set_title("2. Metabolic Cost & Brain Hemodynamics", fontsize=14)
        ax2.legend(loc='upper left')

        ax3 = plt.subplot(3, 1, 3)
        df_melt = self.df_final.melt(id_vars='Phase', value_vars=['NCI_Z', 'PE_Z', 'F_Z', 'HEMO_Z'], var_name='Metric', value_name='Z-Score')
        sns.boxplot(data=df_melt, x='Phase', y='Z-Score', hue='Metric', palette='viridis', ax=ax3, showfliers=False)
        ax3.axhline(0, color='black')
        ax3.set_title("3. Statistical Distribution", fontsize=14)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"SAIM_Result_{self.subject_id}_v9.1.png")
        print(f"Graph saved: SAIM_Result_{self.subject_id}_v9.1.png")
        
        self.df_final.to_csv(f"SAIM_Data_{self.subject_id}_v9.1.csv", index=False)

        # --- OUTPUT: Numerical Summary ---
        print("\n=== Numerical Summary (Mean Z-Score per Phase) ===")
        summary = self.df_final.groupby('Phase', sort=False)[['NCI_Z', 'PE_Z', 'F_Z', 'HEMO_Z']].mean()
        print(summary)

        print("\n=== NCI Bandwidth (Volatility) ===")
        volatility = self.df_final.groupby('Phase', sort=False)['NCI_Z_Std'].mean()
        print(volatility)

    def _add_boundaries(self, ax, df):
        boundaries = df.groupby('PhaseKey').apply(lambda x: x.index[-1]).tolist()[:-1]
        for b in boundaries: ax.axvline(b, color='gray', linestyle=':', alpha=0.8)

# ==========================================================
#   実行ランチャー
# ==========================================================
def auto_run_subject():
    # [ユーザー設定]
    TARGET_ID = 'S99'       
    GROUP_TYPE = 'Real'     
    DATA_FOLDER = '/content/' 
    # -----------------------
    
    is_sham = (GROUP_TYPE == 'Sham')
    files = {}
    try:
        all_files = os.listdir(DATA_FOLDER)
    except FileNotFoundError:
        print(f"Error: Data folder '{DATA_FOLDER}' not found.")
        return

    phases = {
        '01_Pre': '01_Pre', '02_PostImmed': '02_PostImmed',
        '03_PostStress': '03_PostStress', '04_PostRest': '04_PostRest',
        '05_RescueImmed': '05_RescueImmed', '06_RescuePost': '06_RescuePost'
    }
    
    for key, suffix in phases.items():
        match = [f for f in all_files if TARGET_ID in f and suffix in f]
        if match: files[key] = os.path.join(DATA_FOLDER, match[0])
    
    if not files:
        print(f"No files found for ID: {TARGET_ID}")
        return

    subject_label = f"{TARGET_ID}_{GROUP_TYPE}"
    print(f"\n>>> Starting Analysis for: {subject_label}")
    analyzer = SAIMAnalyzer(subject_id=subject_label, file_map=files, is_sham=is_sham)
    analyzer.run_analysis()

if __name__ == "__main__":
    auto_run_subject()
