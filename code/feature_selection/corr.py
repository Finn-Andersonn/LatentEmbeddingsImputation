# -----------------------------------------------------------------------------
# IMPORTANCE: Initial correlation analysis and feature selection for temporal data
# GOAL: Identify and remove redundant features to create a leaner, more stable feature set for imputation.
#   -> Imputation (Step 4): A smaller, less collinear feature set improves the performance of imputation models.
#   -> Contrastive Learning (Step 5): Requires a well-conditioned covariance matrix for meaningful representations.
#   -> Graph-Based Models (Step 6): Benefits from independent features to construct meaningful edges.
#
# 1) Compute cross-correlations to identify optimal lags for temporal relationships.
# 2) Add lagged features and rolling statistics to capture temporal dynamics.
# 3) Compute Spearman correlations with permutation tests on the expanded feature set.
# 4) Use VIF and hierarchical clustering to remove redundant features.
# 5) Handle non-normality with Spearman correlation and permutation tests.
# 6) Handle NaNs by computing correlations on a pairwise basis and imputing for VIF.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import ccf
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CorrelationAnalyzer:
    '''
    Description:
        Difficult. With all these missing rows we ran into an issue: Should we run imputation first, or should we run correlation analysis first 
        and then impute? Should we use a simple imputation method for correlation analysis and then use advanced methods like autoencoders later?

        Decided (reasons in paper): Perform Correlation Analysis Without Imputation first
        Impute (simply) for VIF and Covariance Metrics (we also use these imputation RMSE's for future comparison to more advanced methods)
        NOTE: NEED TO RERUN THE SIMPLE METHODS ON CHOSEN FEATURE SET

        The Shapiro-Wilk test in here showed that features like fill_pct, temperature_avg, and precipitation are non-normal (p < 0.05), 
        even after log transformation in the initial run.

        To make corr.py more suitable for your temporal data, we incorporate the following:
            -> Lagged Features
            -> Rolling Statistics
            -> Use cross-correlation (instead of Spearman correlation) to account for lagged relationships
            -> original Spearman correlations may be inflated
        So we run spearman correlation after keeping the highest cross-correlatiosn from the time-spatial expanded feature set
    
    Args:
        numeric_features: excluding non-numeric columns
    
    Returns:
        normality test, time-lagged/rolling features, cross-correlations, spearman corr, picked features, VIF test
        several plots.
    '''
    def __init__(self, data_path, numeric_features, output_dir="correlation_plots", n_permutations=1000, max_lags=5, random_state=42):
        self.data_path = data_path
        self.numeric_features = numeric_features
        self.output_dir = output_dir
        self.n_permutations = n_permutations
        self.max_lags = max_lags  # Maximum lags for cross-correlation
        self.random_state = random_state
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess_data(self):
        df = pd.read_csv(self.data_path)
        # Ensure temporal order
        if 'imaging_time' in df.columns:
            df['imaging_time'] = pd.to_datetime(df['imaging_time'], errors='coerce')
            df['imaging_hour'] = df['imaging_time'].dt.hour + df['imaging_time'].dt.minute / 60.0
            df = df.sort_values(['tank_id', 'imaging_time']).reset_index(drop=True)
        if 'season' in df.columns and 'season_numeric' not in df.columns:
            df['season_numeric'] = df['season'].map({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}).astype(float)

        # Convert to numeric
        all_cols = list(set(self.numeric_features + ['imaging_hour', 'season_numeric']))
        for col in all_cols:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Handle inconsistent recordings by weighting based on tank farm frequency
        if 'tank farm' in df.columns:
            farm_counts = df['tank farm'].value_counts()
            df['weight'] = df['tank farm'].map(lambda x: 1 / farm_counts[x])
        else:
            df['weight'] = 1.0

        # Define static features (no rolling stats for these)
        static_features = ['distance_to_nearest_port', 'distance_to_nearest_tank_id']
        dynamic_features = [col for col in self.numeric_features if col not in static_features]

        # Add lagged features
        for col in self.numeric_features:
            if col not in df.columns:
                continue
            for lag in range(1, self.max_lags + 1):
                df[f'{col}_lag{lag}'] = df.groupby('tank_id')[col].shift(lag)

        # Add rolling statistics only for dynamic features
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

        # Update expanded features
        self.expanded_features = self.numeric_features.copy()
        for col in self.numeric_features:
            self.expanded_features.extend([f'{col}_lag{lag}' for lag in range(1, self.max_lags + 1)])
        for col in dynamic_features:
            self.expanded_features.extend([f'{col}_roll_mean', f'{col}_roll_std'])

        self.data = df.copy()
        print(f"Original data shape: {self.data.shape}")
        self.data_complete = self.data.copy()
        print(f"Data shape (no NaN dropping): {self.data_complete.shape}")
        print("\nChecking distributions of key features:")
        if 'fill_pct' in self.data.columns:
            for col in ['fill_pct', 'imaging_hour']:
                if col in self.data.columns:
                    print(f"Distribution of {col}:\n{self.data[col].describe()}")

    def test_normality(self, subset_size=5000):
        normality_results = {}
        for feature in self.expanded_features:
            if feature not in self.data_complete.columns:
                continue
            nonnull_vals = self.data_complete[feature].dropna()
            if len(nonnull_vals) < 3:
                normality_results[feature] = np.nan
                continue
            sample = nonnull_vals.sample(n=min(subset_size, len(nonnull_vals)), random_state=self.random_state)
            stat, p = stats.shapiro(sample)
            normality_results[feature] = p

            # Q-Q plot
            plt.figure(figsize=(6, 4))
            stats.probplot(sample, dist="norm", plot=plt)
            plt.title(f"Q-Q Plot for {feature}")
            plt.savefig(os.path.join(self.output_dir, f"qq_plot_{feature}.png"))
            plt.close()

            print(f"Full dataset {feature} mean: {nonnull_vals.mean():.4f}, std: {nonnull_vals.std():.4f}")
            print(f"Subset {feature} mean: {sample.mean():.4f}, std: {sample.std():.4f}")

        return normality_results

    def compute_cross_correlations(self):
        """
        Compute cross-correlations between fill_pct and other features to identify optimal lags.
        """
        if 'fill_pct' not in self.data_complete.columns:
            print("fill_pct not found in data. Skipping cross-correlation.")
            return {}

        ccf_results = {}
        for feature in self.numeric_features:
            if feature == 'fill_pct' or feature not in self.data_complete.columns:
                continue
            max_ccf = 0
            best_lag = 0
            ccf_vals = []
            # Compute cross-correlation for each tank_id
            for tank_id in self.data_complete['tank_id'].unique():
                tank_data = self.data_complete[self.data_complete['tank_id'] == tank_id]
                x = tank_data['fill_pct'].dropna()
                y = tank_data[feature].dropna()
                if len(x) < self.max_lags + 1 or len(y) < self.max_lags + 1:
                    continue
                # Align indices
                common_idx = x.index.intersection(y.index)
                if len(common_idx) < self.max_lags + 1:
                    continue
                x = x.loc[common_idx]
                y = y.loc[common_idx]
                ccf_result = ccf(x, y, adjusted=False)[:self.max_lags + 1]  # Lags 0 to max_lags
                ccf_vals.extend(ccf_result)

            if not ccf_vals:
                print(f"No valid cross-correlation for {feature}.")
                continue

            ccf_vals = np.array(ccf_vals).reshape(-1, self.max_lags + 1)
            mean_ccf = np.nanmean(ccf_vals, axis=0)
            best_lag = np.argmax(np.abs(mean_ccf))
            max_ccf = mean_ccf[best_lag]
            ccf_results[feature] = {'best_lag': best_lag, 'max_ccf': max_ccf, 'ccf_vals': mean_ccf}

            # Plot cross-correlation
            plt.figure(figsize=(8, 6))
            plt.plot(range(self.max_lags + 1), mean_ccf, marker='o')
            plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
            plt.title(f"Cross-Correlation: fill_pct vs {feature}")
            plt.xlabel("Lag")
            plt.ylabel("Cross-Correlation")
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f"ccf_fill_pct_vs_{feature}.png"))
            plt.close()

        # Add optimal lagged features
        for feature, result in ccf_results.items():
            best_lag = result['best_lag']
            if best_lag > 0:  # Only add lagged feature if lag > 0
                lag_col = f'{feature}_lag{best_lag}'
                if lag_col in self.data_complete.columns:
                    print(f"Keeping {lag_col} (best lag: {best_lag}, max CCF: {result['max_ccf']:.4f})")
                else:
                    print(f"Warning: {lag_col} not found in data.")

        return ccf_results

    def permutation_test_corr(self, x, y, weights=None, method='spearman'):
        rng = np.random.default_rng(self.random_state)

        # For Spearman, compute ranks
        if method == 'spearman':
            x_rank = stats.rankdata(x)
            y_rank = stats.rankdata(y)
            if weights is not None:
                w_sum = np.sum(weights)
                x_mean = np.sum(x_rank * weights) / w_sum
                y_mean = np.sum(y_rank * weights) / w_sum
                cov = np.sum(weights * (x_rank - x_mean) * (y_rank - y_mean)) / w_sum
                var_x = np.sum(weights * (x_rank - x_mean)**2) / w_sum
                var_y = np.sum(weights * (y_rank - y_mean)**2) / w_sum
                if var_x == 0 or var_y == 0:
                    r_obs = np.nan
                else:
                    r_obs = cov / np.sqrt(var_x * var_y)
            else:
                r_obs, _ = stats.pearsonr(x_rank, y_rank)
        else:
            r_obs, _ = stats.pearsonr(x, y)

        # Permutation test
        if self.n_permutations <= 0:
            if method == 'spearman':
                _, p = stats.pearsonr(x_rank, y_rank)
            else:
                _, p = stats.pearsonr(x, y)
            return r_obs, p

        cnt = 0
        for _ in range(self.n_permutations):
            y_perm = rng.permutation(y)
            if method == 'spearman':
                y_perm_rank = stats.rankdata(y_perm)
                if weights is not None:
                    w_sum = np.sum(weights)
                    x_mean = np.sum(x_rank * weights) / w_sum
                    y_mean = np.sum(y_perm_rank * weights) / w_sum
                    cov = np.sum(weights * (x_rank - x_mean) * (y_perm_rank - y_mean)) / w_sum
                    var_x = np.sum(weights * (x_rank - x_mean)**2) / w_sum
                    var_y = np.sum(weights * (y_perm_rank - y_mean)**2) / w_sum
                    if var_x == 0 or var_y == 0:
                        r_perm = np.nan
                    else:
                        r_perm = cov / np.sqrt(var_x * var_y)
                else:
                    r_perm, _ = stats.pearsonr(x_rank, y_perm_rank)
            else:
                r_perm, _ = stats.pearsonr(x, y_perm)
            if abs(r_perm) >= abs(r_obs):
                cnt += 1
        p_perm = cnt / self.n_permutations
        return r_obs, p_perm

    def compute_pairwise_correlations(self, method='spearman'):
        features = self.expanded_features
        k = len(features)
        corr_matrix = np.zeros((k, k))
        p_matrix = np.zeros((k, k))

        for i in range(k):
            for j in range(i + 1, k):
                feat_i = features[i]
                feat_j = features[j]
                data_subset = self.data_complete[[feat_i, feat_j, 'weight']].dropna()
                if len(data_subset) < 2:
                    corr_matrix[i, j] = np.nan
                    p_matrix[i, j] = np.nan
                    continue
                x = data_subset[feat_i].values
                y = data_subset[feat_j].values
                weights = data_subset['weight'].values
                r, p = self.permutation_test_corr(x, y, weights=weights, method=method)
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
                p_matrix[i, j] = p
                p_matrix[j, i] = p

        corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
        p_df = pd.DataFrame(p_matrix, index=features, columns=features)
        print("Correlation matrix:")
        print(corr_df)
        print("p-values:")
        print(p_df)
        return corr_df, p_df

    def compute_significance_bounds(self, q, alpha=0.05):
        t_crit = stats.t.ppf(1 - alpha / 2, df=q - 2)
        bound = t_crit / np.sqrt(t_crit**2 + q - 2)
        return bound

    def cluster_features(self, corr_df):
        print("Correlation matrix before clustering:")
        print(corr_df)
        print("Any non-finite values in corr_df?", corr_df.isna().any().any() or np.isinf(corr_df).any().any())

        corr_matrix = corr_df.fillna(0)
        Z = linkage(1 - abs(corr_matrix), method='average', metric='correlation')
        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=corr_matrix.index, leaf_rotation=45)
        plt.title("Hierarchical Clustering of Features Based on Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_clustering.png"))
        plt.close()

    def recommend_features(self, corr_df, p_df, ccf_results, alpha=0.05, vif_threshold=200):
        # Impute missing values for VIF calculation
        X = self.data_complete[self.expanded_features].copy()
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Compute VIF
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_imputed.columns
        vif_data["VIF"] = [variance_inflation_factor(X_imputed.values, i) for i in range(X_imputed.shape[1])]
        print("\nInitial VIF Results:")
        print(vif_data)

        # Hierarchical clustering
        corr_matrix = corr_df.fillna(0)
        Z = linkage(1 - abs(corr_matrix), method='average', metric='correlation')
        from scipy.cluster.hierarchy import fcluster
        clusters = fcluster(Z, t=0.7, criterion='distance')
        cluster_dict = {feat: cluster for feat, cluster in zip(corr_df.index, clusters)}

        # Prioritize features based on cross-correlation and Spearman correlation
        keep_features = []
        drop_features = []
        for cluster in set(clusters):
            cluster_feats = [feat for feat, c in cluster_dict.items() if c == cluster]
            if len(cluster_feats) == 1:
                keep_features.append(cluster_feats[0])
                continue
            # Always keep fill_pct and its derived features
            fill_pct_feats = [f for f in cluster_feats if 'fill_pct' in f]
            keep_features.extend(fill_pct_feats)
            cluster_feats = [f for f in cluster_feats if f not in fill_pct_feats]
            if not cluster_feats:
                continue

            # Score features based on cross-correlation (for original features) and Spearman correlation
            scores = {}
            for feat in cluster_feats:
                # Extract the base feature name (e.g., 'wti_price_lag2' -> 'wti_price')
                base_feat = feat.split('_lag')[0].split('_roll')[0]
                ccf_score = ccf_results.get(base_feat, {}).get('max_ccf', 0)
                spearman_score = abs(corr_df.loc['fill_pct', feat]) if 'fill_pct' in corr_df.index else 0
                scores[feat] = 0.5 * ccf_score + 0.5 * spearman_score  # Weighted combination

            if not scores:
                continue
            best_feat = max(scores, key=scores.get)
            keep_features.append(best_feat)
            drop_features.extend([f for f in cluster_feats if f != best_feat])

        # Recompute VIF on the reduced feature set
        X_reduced = X_imputed[keep_features].copy()
        vif_data_reduced = pd.DataFrame()
        vif_data_reduced["Feature"] = X_reduced.columns
        vif_data_reduced["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])]
        print("\nVIF After Clustering:")
        print(vif_data_reduced)

        final_keep = []
        final_drop = drop_features.copy()
        for _, row in vif_data_reduced.iterrows():
            feat = row["Feature"]
            vif = row["VIF"]
            if vif > vif_threshold:
                final_drop.append(feat)
            else:
                final_keep.append(feat)

        print(f"\nFinal Recommended Features to Keep (after clustering and VIF < {vif_threshold}):")
        print(final_keep)
        print(f"\nFinal Features to Drop (high correlation within clusters or VIF > {vif_threshold}):")
        print(final_drop)
        return final_keep, final_drop

    def test_imaging_hour_vs_weather(self):
        weather_features = ['temperature_avg', 'precipitation']
        for weather_feat in weather_features:
            if weather_feat not in self.data_complete.columns or 'imaging_hour' not in self.data_complete.columns:
                print(f"Required columns for imaging_hour vs {weather_feat} test are missing.")
                continue

            data_subset = self.data_complete[[weather_feat, 'imaging_hour', 'weight']].dropna()
            if len(data_subset) < 2:
                print(f"Not enough data for imaging_hour vs {weather_feat} test.")
                continue

            x = data_subset['imaging_hour'].values
            y = data_subset[weather_feat].values
            weights = data_subset['weight'].values
            r, p = self.permutation_test_corr(x, y, weights=weights, method='spearman')
            print(f"\nSpearman Correlation (imaging_hour vs {weather_feat}):")
            print(f"Correlation = {r:.4f}, p-value = {p:.4g}")
            if p < 0.05:
                print(f"Significant correlation between imaging_hour and {weather_feat} (p < 0.05).")
            else:
                print(f"No significant correlation between imaging_hour and {weather_feat}.")

            # Scatter plot for visualization
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='imaging_hour', y=weather_feat, data=self.data_complete, alpha=0.5)
            plt.title(f'Imaging Hour vs {weather_feat}')
            plt.xlabel('Imaging Hour')
            plt.ylabel(weather_feat)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"imaging_hour_vs_{weather_feat}.png"))
            plt.close()

    def plot_pairwise_correlations(self, corr_df, p_df, alpha=0.05):
        q = len(self.data_complete)
        bound_05 = self.compute_significance_bounds(q, alpha=0.05)
        bound_01 = self.compute_significance_bounds(q, alpha=0.01)
        print(f"Significance bounds: 5% = {bound_05:.3f}, 1% = {bound_01:.3f}")

        # Apply Bonferroni correction
        n_comparisons = (len(corr_df) * (len(corr_df) - 1)) // 2
        adjusted_alpha = alpha / n_comparisons
        print(f"Bonferroni-adjusted alpha: {adjusted_alpha:.6f}")

        mask = (p_df > adjusted_alpha) | (abs(corr_df) < bound_05)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", square=True, mask=mask, center=0)
        plt.title(f"Pairwise Spearman Correlations (Significant at p < {adjusted_alpha:.6f} and |r| > 5% Bound)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pairwise_correlations.png"))
        plt.close()

    def run_analysis(self):
        self.preprocess_data()

        # Normality test
        normality_results = self.test_normality()
        print("Normality Test (Shapiro-Wilk p-values on subset):")
        for feature, pval in normality_results.items():
            if np.isnan(pval):
                print(f"  {feature}: not enough data or missing.")
            else:
                msg = "Non-normal" if pval < 0.05 else "Normal"
                print(f"  {feature}: p = {pval:.4g} ({msg})")

        # Compute cross-correlations to identify optimal lags
        print("\nComputing cross-correlations...")
        ccf_results = self.compute_cross_correlations()

        # Compute pairwise Spearman correlations on the expanded feature set
        method = 'spearman'
        print(f"\nUsing {method} correlation method. n_permutations={self.n_permutations}")
        print("\nComputing pairwise correlations...")
        corr_df, p_df = self.compute_pairwise_correlations(method=method)
        self.plot_pairwise_correlations(corr_df, p_df)

        # Feature clustering
        print("\nClustering features based on correlations...")
        self.cluster_features(corr_df)

        # Test imaging_hour vs weather features
        self.test_imaging_hour_vs_weather()

        # Feature recommendation
        keep_features, drop_features = self.recommend_features(corr_df, p_df, ccf_results, vif_threshold=100)

        print("\nAnalysis complete.")
        return keep_features, drop_features

if __name__ == "__main__":
    numeric_features = [
        "fill_pct", "temperature_avg", "temperature_min",
        "temperature_max", "precipitation", "wind_speed", "wind_dir",
        "pressure",
        "distance_to_nearest_port", "distance_to_nearest_tank_id",
        "wti_price", "brent_price", "wti_brent_spread", "season_numeric"
    ]
    #selected_features = ['fill_pct', 'distance_to_nearest_tank_id', 'precipitation', 'wind_dir', 'distance_to_nearest_port', 'wti_price']

    analyzer = CorrelationAnalyzer(
        data_path="/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/missingness/complete_panel_with_features_updated.csv",
        numeric_features=numeric_features,
        output_dir="/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/corr/",
        n_permutations=20,
        max_lags=3,
        random_state=42
    )

    # Run the analysis
    keep_features, drop_features = analyzer.run_analysis()

    # Save the recommended features
    with open(os.path.join(analyzer.output_dir, "recommended_features.txt"), "w") as f:
        f.write("Recommended Features to Keep:\n")
        f.write("\n".join(keep_features))
        f.write("\n\nFeatures to Drop:\n")
        f.write("\n".join(drop_features))