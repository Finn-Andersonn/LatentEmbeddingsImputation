# ---------------------------------
# Need to ensure covariance estimate is accurate -> going to use later
# Run this over imputed dataset
#
# ---------------------------------


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#NOTE: THIS IS TO BE RUN BY THE BEST DATASET FROM MEDIUM_IMP.PY

class StabilityAnalyzer:
    '''
    Description:
        Check for Frobenius norm, condition number, regression error
        Trying to confirm the features I picked from corr.py and mutual_info.py are satisfactory
    
    Args:
        numeric_features: features selected after corr.py, corroborated with mutual_info.py
        df: restructured pandas dataframe

    Returns:
       frobenius_norms, condition_numbers, regression_errors_fill, wasserstein_dists, kld_p_vals
    '''
    def __init__(self, data_path, numeric_features, output_dir="stability_plots", random_state=42):
        self.data_path = data_path
        self.numeric_features = numeric_features
        self.output_dir = output_dir
        self.random_state = random_state
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        df = pd.read_csv(self.data_path)
        if 'imaging_time' in df.columns:
            df['imaging_time'] = pd.to_datetime(df['imaging_time'], errors='coerce')
            df['imaging_hour'] = df['imaging_time'].dt.hour + df['imaging_time'].dt.minute / 60.0
        if 'season' in df.columns and 'season_numeric' not in df.columns:
            df['season_numeric'] = df['season'].map({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}).astype(float)

        all_cols = list(set(self.numeric_features + ['imaging_hour', 'season_numeric']))
        for col in all_cols:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

        # handle inconsistent recordings by weighting based on tank farm frequency
        if 'tank farm' in df.columns:
            farm_counts = df['tank farm'].value_counts()
            df['weight'] = df['tank farm'].map(lambda x: 1 / farm_counts[x])
        else:
            df['weight'] = 1.0

        self.data = df.copy()
        print(f"Loaded data shape: {self.data.shape}")
        self.data_complete = self.data.copy()
        print(f"Data shape (post-load): {self.data_complete.shape}")
        print("\nChecking distributions of key features:")
        for col in ['fill_pct', 'imaging_hour']:
            if col in self.data.columns:
                print(f"Distribution of {col}:\n{self.data[col].describe()}")

    def compute_covariance_metrics(self, sample_sizes, features=None):
        if features is None:
            features = self.numeric_features
        p = len(features)
        frobenius_norms = {p_val: [] for p_val in [50, 100, 500] if p_val <= p}
        condition_numbers = []
        regression_errors_fill = []
        wasserstein_dists = []
        kld_p_vals = []

        # Frobenius norm for different p
        for p_val in frobenius_norms.keys():
            selected_features = features[:p_val]
            X = self.data_complete[selected_features].copy()
            true_cov = X.cov()
            true_cov += np.eye(true_cov.shape[0]) * 1e-6  # Regularization
            for q in sample_sizes:
                if q > len(self.data_complete):
                    q = len(self.data_complete)
                sample_data = self.data_complete.sample(n=q, random_state=self.random_state)
                sample_cov = sample_data[selected_features].cov()
                sample_cov += np.eye(sample_cov.shape[0]) * 1e-6
                frob = np.linalg.norm(true_cov - sample_cov, 'fro')
                frobenius_norms[p_val].append(frob)

        # Other metrics with the actual p
        X = self.data_complete[features].copy()
        true_cov = X.cov()
        true_cov += np.eye(true_cov.shape[0]) * 1e-6
        true_dist = {col: X[col].values for col in features}

        for q in sample_sizes:
            if q > len(self.data_complete):
                q = len(self.data_complete)
            sample_data = self.data_complete.sample(n=q, random_state=self.random_state)
            sample_cov = sample_data[features].cov()
            sample_cov += np.eye(sample_cov.shape[0]) * 1e-6

            # Condition number
            cond_num = np.linalg.cond(sample_cov)
            condition_numbers.append(cond_num)

            # Regression errors for fill_pct
            fill_col = next((c for c in features if 'fill_pct' in c), None)
            if fill_col:
                drop_cols = [fill_col]
                X_sample = sample_data.drop(columns=drop_cols + ['weight'], errors='ignore')
                y_sample = sample_data[fill_col]
                X_full = self.data_complete.drop(columns=drop_cols + ['weight'], errors='ignore')
                y_full = self.data_complete[fill_col]
                if len(X_sample) > 5:
                    model = LinearRegression()
                    model.fit(X_sample, y_sample)
                    y_pred = model.predict(X_full)
                    mse_fill = mean_squared_error(y_full, y_pred)
                    regression_errors_fill.append(mse_fill)
                else:
                    regression_errors_fill.append(np.nan)
            else:
                regression_errors_fill.append(np.nan)

            # Wasserstein distance
            w_dist = 0
            for col in features:
                true_dist_col = true_dist[col]
                sample_dist = sample_data[col].values
                w_dist += wasserstein_distance(true_dist_col, sample_dist)
            w_dist /= len(features)
            wasserstein_dists.append(w_dist)

            # KLD/p (for reference, but we’ll rely on Wasserstein due to non-normality => confirmed in corr.py)
            kld = 0
            for col in features:
                true_dist_col = true_dist[col]
                sample_dist = sample_data[col].values
                hist_true, bins = np.histogram(true_dist_col, bins=30, density=True)
                hist_sample, _ = np.histogram(sample_dist, bins=bins, density=True)
                hist_true = np.where(hist_true == 0, 1e-10, hist_true)
                hist_sample = np.where(hist_sample == 0, 1e-10, hist_sample)
                kld += np.sum(hist_true * np.log(hist_true / hist_sample)) * (bins[1] - bins[0])
            kld_p = kld / p
            kld_p_vals.append(kld_p)

        return frobenius_norms, condition_numbers, regression_errors_fill, wasserstein_dists, kld_p_vals

    def plot_covariance_metrics(self, sample_sizes, frobenius_norms, condition_numbers,
                                regression_errors_fill, wasserstein_dists, kld_p_vals):
        p = len(self.numeric_features)
        qp_ratios = [s / p for s in sample_sizes]

        # Condition number
        plt.figure(figsize=(10, 6))
        plt.plot(qp_ratios, condition_numbers, marker='o', label='Condition Number')
        plt.axhline(y=1e4, color='red', linestyle='--', label='Threshold 1e4')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('q/p Ratio')
        plt.ylabel('Condition Number')
        plt.title('Condition Number vs q/p')
        plt.legend()
        plt.grid(True, which='both', ls='--')
        valid_cond_nums = [x for x in condition_numbers if not np.isnan(x) and not np.isinf(x)]
        if valid_cond_nums:
            plt.ylim(min(valid_cond_nums) * 0.9, max(valid_cond_nums) * 1.1)
        plt.savefig(os.path.join(self.output_dir, "condition_number_vs_qp.png"))
        plt.close()

        # Regression errors
        if regression_errors_fill:
            plt.figure(figsize=(10, 6))
            plt.plot(qp_ratios, regression_errors_fill, marker='o', label='Fill_pct Error')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('q/p Ratio')
            plt.ylabel('MSE')
            plt.title('Out-of-sample Regression Error vs q/p')
            plt.legend()
            plt.grid(True, which='both', ls='--')
            valid_errors = [x for x in regression_errors_fill if not np.isnan(x) and not np.isinf(x)]
            if valid_errors:
                plt.ylim(min(valid_errors) * 0.9, max(valid_errors) * 1.1)
            plt.savefig(os.path.join(self.output_dir, "regression_error_vs_qp.png"))
            plt.close()

        # Wasserstein distance
        plt.figure(figsize=(10, 6))
        plt.plot(qp_ratios, wasserstein_dists, marker='o', label='Wasserstein Distance')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('q/p Ratio')
        plt.ylabel('Wasserstein Distance')
        plt.title('Wasserstein Distance vs q/p')
        plt.legend()
        plt.grid(True, which='both', ls='--')
        valid_w_dists = [x for x in wasserstein_dists if not np.isnan(x) and not np.isinf(x)]
        if valid_w_dists:
            plt.ylim(min(valid_w_dists) * 0.9, max(valid_w_dists) * 1.1)
        plt.savefig(os.path.join(self.output_dir, "wasserstein_vs_qp.png"))
        plt.close()

        # KLD/p
        plt.figure(figsize=(10, 6))
        plt.plot(qp_ratios, kld_p_vals, marker='o', label='KLD/p')
        q_vals = np.array(sample_sizes)
        kld_theoretical = 0.251 / q_vals
        plt.plot(qp_ratios, kld_theoretical, linestyle='--', color='purple', label='0.251/q')
        kld_uncorrelated = np.mean([k for k, q in zip(kld_p_vals, qp_ratios) if q > 1000])
        plt.axhline(y=kld_uncorrelated, color='orange', linestyle='--', label='Uncorrelated Variables')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('q/p Ratio')
        plt.ylabel('KLD/p')
        plt.title('Kullback-Leibler Divergence per Unit of Density vs q/p')
        plt.legend()
        plt.grid(True, which='both', ls='--')
        valid_kld_p = [x for x in kld_p_vals if not np.isnan(x) and not np.isinf(x)]
        if valid_kld_p:
            plt.ylim(min(valid_kld_p) * 0.9, max(valid_kld_p) * 1.1)
        plt.savefig(os.path.join(self.output_dir, "kld_p_vs_qp.png"))
        plt.close()

    def run_analysis(self, sample_sizes=None):
        self.load_data()

        if sample_sizes is None:
            sample_sizes = list(range(10, 200, 10))
        sample_sizes = [s for s in sample_sizes if s <= len(self.data_complete)]

        # covariance metrics
        frob, cond_nums, err_fill, w_dists, kld_p = self.compute_covariance_metrics(sample_sizes, features=self.numeric_features)
        self.plot_covariance_metrics(sample_sizes, frob, cond_nums, err_fill, w_dists, kld_p)

        print("\nStability analysis complete.")

if __name__ == "__main__":
    # Use the features selected from corr.py
    numeric_features = [
        "fill_pct",
        "temperature_avg",
        "precipitation",
        "wind_speed",
        "distance_to_nearest_port",
        "wti_price",
        "season_numeric"
    ]  # Update this list based on corr.py output

    analyzer = StabilityAnalyzer(
        data_path="/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/missingness/complete_panel_with_features_updated.csv",  # Update with the path to your imputed dataset
        numeric_features=numeric_features,
        output_dir="/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/stability/",
        random_state=42
    )

    analyzer.run_analysis()