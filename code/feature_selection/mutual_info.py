# ---------------------------------------------
# While i.i.d. is violate, we mitigate the issue by adding in lagged and rolling values
# but to assure we don't only assess linear (Pearson) or monotonic (Spearman) relations
# we include mutual information (more of a confirmation of the features that followed the
# cross-corrletaion/Spearman process)
#
# To avoid confusion:
#   -> MI(X, Y) = 1.8 means that knowing X reduces the uncertainty about Y by 1.8499 bits (on average).
#   -> MI(X, Y) = 0.0986 indicates a much smaller reduction in uncertainty, suggesting a weaker relationship.
#
# Uses sklearn.feature_selection.mutual_info_regression, which estimates MI for continuous variables using a k-nearest neighbors (k-NN) approach.
#   -> we use Bootstrapping (50 iterations) to provide confidence intervals to account for variability in the estimate.
#   -> The script computes MI between all pairs of features and recommends features with high MI with fill_pct (top 50%).

# KEY: mutual_info_regression assumes that observations are independent and identically distributed (i.i.d.). 
#   -> This means each row in the dataset (a (tank_id, period) pair) should be independent of other rows.
#   -> Observations for the same tank_id across different periods are likely correlated. For example, fill_pct for a tank on day t might be similar to day t+1 due to slow changes in tank levels.
#   -> This violates the independence assumption because the effective sample size is reduced (observations are not truly independent).
# Additionally, spatial correlation exists between tanks in the same region.
# ---------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import os

# NOTE: WE NEED TO BE AWARE OF INDEPENDENCE ASSUMPTION

class MutualInfoAnalyzer:
    def __init__(self, data_path, numeric_features, output_dir='mi_plots', transform_data=False, n_bootstraps=50, random_state=42, max_lags=3):
        self.data_path = data_path
        self.numeric_features = numeric_features
        self.output_dir = output_dir
        self.transform_data = transform_data
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        self.max_lags = max_lags  # Added to control the number of lags
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_and_preprocess(self):
        '''
        Description:
            Same general function as corr.py, etc.
        
        '''
        df = pd.read_csv(self.data_path)
        df['imaging_time'] = pd.to_datetime(df['imaging_time'], errors='coerce')
        df['imaging_hour'] = df['imaging_time'].dt.hour + df['imaging_time'].dt.minute / 60

        if 'season' in df.columns and 'season_numeric' not in df.columns:
            df['season_numeric'] = df['season'].map({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}).astype(float)

        if 'tank_id' in df.columns and 'imaging_time' in df.columns:
            df = df.sort_values(['tank_id', 'imaging_time']).reset_index(drop=True)

        # define static features (!!no rolling stats for these)
        static_features = ['distance_to_nearest_port', 'distance_to_nearest_tank_id']
        dynamic_features = [col for col in self.numeric_features if col not in static_features]

        # add lagged features
        for col in self.numeric_features:
            if col not in df.columns:
                continue
            for lag in range(1, self.max_lags + 1):
                df[f'{col}_lag{lag}'] = df.groupby('tank_id')[col].shift(lag)

        # rolling statistics only for dynamic features
        for col in dynamic_features:
            if col not in df.columns:
                continue
            df[f'{col}_roll_mean'] = (df.groupby('tank_id')[col]
                                      .rolling(window=7, min_periods=1)
                                      .mean()
                                      .reset_index(level=0, drop=True))
            df[f'{col}_roll_std'] = (df.groupby('tank_id')[col]
                                     .rolling(window=7, min_periods=1)
                                     .std()
                                     .reset_index(level=0, drop=True))

        # update expanded features
        expanded_features = self.numeric_features.copy()
        for col in self.numeric_features:
            expanded_features.extend([f'{col}_lag{lag}' for lag in range(1, self.max_lags + 1)])
        for col in dynamic_features:
            expanded_features.extend([f'{col}_roll_mean', f'{col}_roll_std'])
        self.numeric_features = expanded_features

        # convert to numeric
        all_features = list(set(self.numeric_features + ['days_since_eia_report']))
        for col in all_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Optional log transform -> didnt help previousy, hence I made toggle
        if self.transform_data:
            for col in all_features:
                if col in df.columns:
                    min_val = df[col].min()
                    if pd.isna(min_val):
                        continue
                    if min_val <= 0:
                        shift = abs(min_val) + 1e-6
                        df[f'{col}_log'] = np.log(df[col] + shift)
                    else:
                        df[f'{col}_log'] = np.log(df[col] + 1e-6)
            self.numeric_features = [f'{f}_log' for f in self.numeric_features if f'{f}_log' in df.columns]
            all_log_features = self.numeric_features
            if 'days_since_eia_report_log' in df.columns:
                all_log_features.append('days_since_eia_report_log')
        else:
            all_log_features = [f for f in all_features if f in df.columns]

        # standardize features -> Why this method? Addressed in paper
        for col in all_log_features:
            if col not in df.columns:
                continue
            df[col] = (df[col] - df[col].mean()) / df[col].std()

        # drop NaNs (reduces rows? double-check)
        self.df = df.dropna(subset=all_log_features)
        return self.df

    def compute_mi_with_bootstrap(self, x, y, discrete_x=False):
        rng = np.random.default_rng(self.random_state)
        mi_vals = []
        n_samples = len(x)
        for i in range(self.n_bootstraps):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            mi = mutual_info_regression(x_boot.reshape(-1, 1), y_boot, discrete_features=discrete_x, random_state=self.random_state)[0]
            mi_vals.append(mi)
            if (i + 1) % 10 == 0:
                print(f"Bootstrap {i + 1}/{self.n_bootstraps}: MI = {mi:.4f}")
        mi_mean = np.mean(mi_vals)
        mi_ci = np.percentile(mi_vals, [2.5, 97.5])
        return mi_mean, mi_ci

    def run_analysis(self):
        df = self.load_and_preprocess()
        features = self.numeric_features
        if 'days_since_eia_report' in df.columns:
            features.append('days_since_eia_report')
        
        # Compute full MI matrix
        mi_matrix = np.zeros((len(features), len(features)))
        mi_ci_matrix = np.zeros((len(features), len(features), 2))
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i == j:
                    mi_matrix[i, j] = np.nan
                    continue
                x = df[feat1].values
                y = df[feat2].values
                discrete_x = 'days_since_eia_report' in feat1
                mi_val, mi_ci = self.compute_mi_with_bootstrap(x, y, discrete_x=discrete_x)
                mi_matrix[i, j] = mi_val
                mi_ci_matrix[i, j, 0] = mi_ci[0]
                mi_ci_matrix[i, j, 1] = mi_ci[1]
                print(f"MI({feat1}, {feat2}) = {mi_val:.4f} [{mi_ci[0]:.4f}, {mi_ci[1]:.4f}]")
                print(f"Processed {j + 1} features for {feat1}")

        # MI matrix as a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(mi_matrix, annot=False, cmap='Blues', xticklabels=features, yticklabels=features)
        plt.title('Mutual Information Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "mi_matrix.png"))
        plt.close()

        # scatter plots for fill_pct vs other features
        fill_pct_col = next((f for f in features if 'fill_pct' in f), None)
        if fill_pct_col:
            for feat in features:
                if feat == fill_pct_col:
                    continue
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=df[feat], y=df[fill_pct_col])
                plt.xlabel(feat)
                plt.ylabel(fill_pct_col)
                plt.title(f'Scatter Plot: {fill_pct_col} vs {feat}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"scatter_{fill_pct_col}_vs_{feat}.png"))
                plt.close()

        # feature recommendation based on MI with fill_pct
        if fill_pct_col:
            mi_vals = mi_matrix[features.index(fill_pct_col), :]
            mi_vals = [v if not np.isnan(v) else -np.inf for v in mi_vals]
            threshold = np.percentile(mi_vals, 50)  # Keep top 50%
            recommended_keep = set()
            for feat, mi in zip(features, mi_vals):
                if mi >= threshold:
                    # Remove '_lagX', '_roll_mean', '_roll_std' suffixes for recommendation
                    base_feat = feat
                    for suffix in [f'_lag{i}' for i in range(1, self.max_lags + 1)] + ['_roll_mean', '_roll_std']:
                        base_feat = base_feat.replace(suffix, '')
                    recommended_keep.add(base_feat)
            print("\nRecommended features based on MI with fill_pct (top 50%):")
            print(recommended_keep)

        return mi_matrix, mi_ci_matrix

if __name__ == "__main__":
    numeric_features = [
        "fill_pct", "temperature_avg", "temperature_min",
        "temperature_max", "precipitation", "wind_speed", "wind_dir",
        "pressure", "distance_to_nearest_port", "distance_to_nearest_tank_id",
        "wti_price", "brent_price", "wti_brent_spread", "season_numeric"
    ]
    analyzer = MutualInfoAnalyzer(
        data_path="/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/missingness/complete_panel_with_features_updated.csv",
        numeric_features=numeric_features,
        output_dir="/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/mi_plots/",
        transform_data=False,  # False to avoid log transformation
        n_bootstraps=10,  # bootstrap iterations
        random_state=42,
        max_lags=3  # Match corr.py
    )
    mi_matrix, mi_ci_matrix = analyzer.run_analysis()