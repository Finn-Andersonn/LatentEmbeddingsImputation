# ----------------------------------------------
# IMPORTANCE: Additional feature justification
# 1) Which features to keep?
# 
# 2) We need to test for correlation: 
#   -> between the imaging time and the release of the EIA data to see if the timings are motivated by the reporting period
#   -> between the imaging time and the weather data to see if the imaging is influenced by weather conditions
#      which would explain the photos being taken solely before 12pm
#
# 3) Take into account spurrious correlations
#   -> For uncorrelated normally distributed variables, the sample correlation coefficient from a sample size of q has 
#      a t-distribution with q-2 degrees of freedom
#   -> We'll plot two lines indicating the boundaries of the regions where the sample correlation coefficients with absolute values of correlation
#      larger than the bounds at 5% and 1% significance levels. The significance region increases with the sample size
# 2.5) Before that we need to ensure our covariance estimate is accurate
#   -> Maybe we need normality for the sample correlation coefficient to be normally distributed
#   -> Check the Frobenius norm of the difference between the true covariance and its sample estimate vs sample size
#   -> law of large numbers in here
#   -> Wary of curse of dimensionality for inverse (q/p) -> Check the condition number!!
#   -> Plot the condition number of the covariance matrix vs (q/p), the relative regression error on out-of-sample data vs (q/p), and the KLD/p vs q/p 
#      (include dotted lines for a model which assumes uncorrelated variables)
# ----------------------------------------------

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class CorrelationAnalyzer:
    def __init__(self, data_path, target_features, numeric_features, output_dir="correlation_plots"):
        """
        Initialize the CorrelationAnalyzer.

        Parameters:
        - data_path (str): Path to the dataset CSV file.
        - target_features (list): Features to test correlations against (e.g., ['imaging_time', 'fill_pct']).
        - numeric_features (list): Numeric features for correlation and covariance analysis.
        - output_dir (str): Directory to save plots.
        """
        self.data = pd.read_csv(data_path)
        self.target_features = target_features
        self.numeric_features = numeric_features
        self.output_dir = output_dir
        self.q = len(self.data)  # Sample size
        self.p = len(self.numeric_features)  # Number of features
        self.qp_ratio = self.q / self.p

    def preprocess_data(self):
        """
        Preprocess the dataset: convert imaging_time to numeric, handle missing values.
        """
        # Convert imaging_time to numeric (hours since midnight)
        self.data['imaging_time'] = pd.to_datetime(self.data['imaging_time'])
        self.data['imaging_hour'] = self.data['imaging_time'].dt.hour + self.data['imaging_time'].dt.minute / 60

        # Handle missing values for correlation (pairwise deletion will be used in corr())
        # For covariance analysis, we'll use a subset with no missing values
        self.data_complete = self.data[self.numeric_features].dropna()

    def test_normality(self):
        """
        Test normality of numeric features using Shapiro-Wilk test.
        Returns a dictionary of p-values.
        """
        normality_results = {}
        for feature in self.numeric_features:
            stat, p = stats.shapiro(self.data[feature].dropna())
            normality_results[feature] = p
        return normality_results

    def compute_correlations(self, feature1, feature2, method='spearman'):
        """
        Compute correlation between two features, handling missing values with pairwise deletion.

        Parameters:
        - feature1, feature2 (str): Features to correlate.
        - method (str): 'pearson' or 'spearman'.

        Returns:
        - r (float): Correlation coefficient.
        - p (float): P-value.
        """
        data_subset = self.data[[feature1, feature2]].dropna()
        if len(data_subset) < 2:
            return np.nan, np.nan
        r, p = stats.spearmanr(data_subset[feature1], data_subset[feature2]) if method == 'spearman' else stats.pearsonr(data_subset[feature1], data_subset[feature2])
        return r, p

    def compute_significance_bounds(self, alpha=0.05):
        """
        Compute the significance bounds for correlation coefficients assuming uncorrelated variables.

        Parameters:
        - alpha (float): Significance level (e.g., 0.05 for 5%).

        Returns:
        - bound (float): Absolute value of the correlation coefficient at the significance level.
        """
        t_crit = stats.t.ppf(1 - alpha / 2, df=self.q - 2)
        bound = t_crit / np.sqrt(t_crit**2 + self.q - 2)
        return bound

    def plot_correlations(self, correlations, title, filename):
        """
        Plot correlation coefficients with significance bounds.

        Parameters:
        - correlations (dict): Dictionary of (feature, (r, p)) pairs.
        - title (str): Plot title.
        - filename (str): File to save the plot.
        """
        features = list(correlations.keys())
        r_values = [correlations[f][0] for f in features]
        p_values = [correlations[f][1] for f in features]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(features, r_values, color='skyblue')
        for bar, p in zip(bars, p_values):
            color = 'green' if p < 0.05 else 'red'
            bar.set_edgecolor(color)
            bar.set_linewidth(2)

        # Add significance bounds
        bound_5 = self.compute_significance_bounds(alpha=0.05)
        bound_1 = self.compute_significance_bounds(alpha=0.01)
        plt.axhline(y=bound_5, color='orange', linestyle='--', label='5% Significance Bound')
        plt.axhline(y=-bound_5, color='orange', linestyle='--')
        plt.axhline(y=bound_1, color='red', linestyle='--', label='1% Significance Bound')
        plt.axhline(y=-bound_1, color='red', linestyle='--')

        plt.title(title)
        plt.xlabel('Feature')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}")
        plt.close()

    def compute_covariance_metrics(self, sample_sizes):
        """
        Compute covariance metrics (Frobenius norm, condition number, regression error, KLD) vs. sample size.

        Parameters:
        - sample_sizes (list): List of sample sizes to test.

        Returns:
        - frobenius_norms, condition_numbers, regression_errors, klds: Lists of metrics.
        """
        frobenius_norms = []
        condition_numbers = []
        regression_errors = []
        klds = []

        # True covariance (from complete data)
        true_cov = self.data_complete[self.numeric_features].cov()

        for q in sample_sizes:
            # Subsample data
            sample_data = self.data_complete.sample(n=q, random_state=42)
            sample_cov = sample_data[self.numeric_features].cov()

            # Frobenius norm
            frobenius_norm = np.linalg.norm(true_cov - sample_cov, 'fro')
            frobenius_norms.append(frobenius_norm)

            # Condition number
            condition_number = np.linalg.cond(sample_cov)
            condition_numbers.append(condition_number)

            # Regression error (predict fill_pct)
            X = sample_data.drop(columns=['fill_pct']).values
            y = sample_data['fill_pct'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            regression_errors.append(mse)

            # KL Divergence (assuming multivariate normal)
            # For uncorrelated model, assume diagonal covariance with same variances
            sample_mean = sample_data[self.numeric_features].mean().values
            true_mean = self.data_complete[self.numeric_features].mean().values
            sample_cov_diag = np.diag(np.diag(sample_cov))
            try:
                kld = 0.5 * (np.log(np.linalg.det(sample_cov_diag) / np.linalg.det(true_cov)) +
                             np.trace(np.linalg.inv(sample_cov_diag) @ true_cov) +
                             (true_mean - sample_mean).T @ np.linalg.inv(sample_cov_diag) @ (true_mean - sample_mean) -
                             self.p)
                klds.append(kld / self.p)
            except:
                klds.append(np.nan)

        return frobenius_norms, condition_numbers, regression_errors, klds

    def plot_covariance_metrics(self, sample_sizes, frobenius_norms, condition_numbers, regression_errors, klds):
        """
        Plot covariance metrics vs. q/p ratio.
        """
        qp_ratios = [q / self.p for q in sample_sizes]

        # Frobenius Norm
        plt.figure(figsize=(8, 6))
        plt.plot(qp_ratios, frobenius_norms, marker='o', label='Frobenius Norm')
        plt.xlabel('q/p Ratio')
        plt.ylabel('Frobenius Norm')
        plt.title('Frobenius Norm of Covariance Difference vs. q/p')
        plt.legend()
        plt.savefig(f"{self.output_dir}/frobenius_norm_vs_qp.png")
        plt.close()

        # Condition Number
        plt.figure(figsize=(8, 6))
        plt.plot(qp_ratios, condition_numbers, marker='o', label='Condition Number')
        plt.axhline(y=1e4, color='red', linestyle='--', label='Threshold (10^4)')
        plt.xlabel('q/p Ratio')
        plt.ylabel('Condition Number')
        plt.title('Condition Number vs. q/p')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f"{self.output_dir}/condition_number_vs_qp.png")
        plt.close()

        # Regression Error
        plt.figure(figsize=(8, 6))
        plt.plot(qp_ratios, regression_errors, marker='o', label='Regression Error')
        plt.xlabel('q/p Ratio')
        plt.ylabel('MSE')
        plt.title('Regression Error vs. q/p')
        plt.legend()
        plt.savefig(f"{self.output_dir}/regression_error_vs_qp.png")
        plt.close()

        # KL Divergence
        plt.figure(figsize=(8, 6))
        plt.plot(qp_ratios, klds, marker='o', label='KLD/p')
        plt.axhline(y=0, color='black', linestyle='--', label='Uncorrelated Model')
        plt.xlabel('q/p Ratio')
        plt.ylabel('KLD/p')
        plt.title('KL Divergence per Feature vs. q/p')
        plt.legend()
        plt.savefig(f"{self.output_dir}/kld_vs_qp.png")
        plt.close()

    def run_analysis(self):
        """
        Run the full correlation and covariance analysis.
        """
        # Preprocess data
        self.preprocess_data()

        # Test normality
        normality_results = self.test_normality()
        print("Normality Test (Shapiro-Wilk p-values):")
        for feature, p in normality_results.items():
            print(f"{feature}: p = {p:.4f} {'(Non-normal)' if p < 0.05 else '(Normal)'}")

        # Use Spearman correlation if any feature is non-normal
        method = 'spearman' if any(p < 0.05 for p in normality_results.values()) else 'pearson'
        print(f"Using {method} correlation due to normality results.")

        # Test correlations with imaging_time
        imaging_correlations = {}
        for feature in ['days_since_eia_report', 'temperature', 'dwpt', 'rhum', 'wspd']:
            r, p = self.compute_correlations('imaging_hour', feature, method=method)
            imaging_correlations[feature] = (r, p)
            print(f"Correlation between imaging_hour and {feature}: r = {r:.4f}, p = {p:.4f}")

        self.plot_correlations(imaging_correlations, "Correlation with Imaging Hour", "imaging_hour_correlations.png")

        # Test correlations with fill_pct
        fill_correlations = {}
        for feature in ['season', 'wti_price', 'brent_price', 'wti_brent_spread', 'distance_to_nearest_port']:
            # Convert season to numeric for correlation (e.g., Winter=1, Spring=2, etc.)
            if feature == 'season':
                self.data['season_numeric'] = self.data['season'].map({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4})
                r, p = self.compute_correlations('fill_pct', 'season_numeric', method=method)
            else:
                r, p = self.compute_correlations('fill_pct', feature, method=method)
            fill_correlations[feature] = (r, p)
            print(f"Correlation between fill_pct and {feature}: r = {r:.4f}, p = {p:.4f}")

        self.plot_correlations(fill_correlations, "Correlation with Fill Percentage", "fill_pct_correlations.png")

        # Covariance analysis
        sample_sizes = [100, 500, 1000, 5000, 10000, len(self.data_complete)]
        frobenius_norms, condition_numbers, regression_errors, klds = self.compute_covariance_metrics(sample_sizes)
        self.plot_covariance_metrics(sample_sizes, frobenius_norms, condition_numbers, regression_errors, klds)

        # Recommend features to keep
        self.recommend_features(imaging_correlations, fill_correlations)

    def recommend_features(self, imaging_correlations, fill_correlations):
        """
        Recommend features to keep based on correlation results.
        """
        print("\nFeature Retention Recommendations:")
        keep_features = ['tank_id', 'farm_type', 'region', 'imaging_time', 'image_time_round_day',
                         'max_vol', 'fill_pct', 'tank volume', 'Location', 'tank farm', 'lat', 'lon']
        drop_features = ['scene_id', 'provider_scene_id', 'scene_source_id', 'tank_id.1']

        # Weather features
        weather_features = ['temperature', 'dwpt', 'rhum', 'wspd']
        for feature in weather_features:
            r, p = imaging_correlations[feature]
            if p < 0.05 and abs(r) > 0.1:
                print(f"Keep {feature}: Significant correlation with imaging_hour (r = {r:.4f}, p = {p:.4f})")
                keep_features.append(feature)
            else:
                print(f"Drop {feature}: Insignificant correlation with imaging_hour (r = {r:.4f}, p = {p:.4f})")
                drop_features.append(feature)

        # EIA features
        eia_features = ['days_since_eia_report', 'is_eia_report_week', 'eia_report_cycle']
        for feature in eia_features:
            if feature in imaging_correlations:
                r, p = imaging_correlations[feature]
                if p < 0.05 and abs(r) > 0.1:
                    print(f"Keep {feature}: Significant correlation with imaging_hour (r = {r:.4f}, p = {p:.4f})")
                    keep_features.append(feature)
                else:
                    print(f"Drop {feature}: Insignificant correlation with imaging_hour (r = {r:.4f}, p = {p:.4f})")
                    drop_features.append(feature)

        # Fill_pct correlated features
        for feature in fill_correlations:
            r, p = fill_correlations[feature]
            if p < 0.05 and abs(r) > 0.1:
                print(f"Keep {feature}: Significant correlation with fill_pct (r = {r:.4f}, p = {p:.4f})")
                keep_features.append(feature)
            else:
                print(f"Drop {feature}: Insignificant correlation with fill_pct (r = {r:.4f}, p = {p:.4f})")
                drop_features.append(feature)

        # Always keep spatial and temporal features for graph and RAG
        keep_features.extend(['distance_to_nearest_port', 'distance_to_major_city', 'distance_to_coast', 'is_coastal',
                              'day_of_week', 'month', 'season'])

        print("\nFinal Features to Keep:", keep_features)
        print("Features to Drop:", drop_features)

if __name__ == "__main__":
    target_features = ['imaging_hour', 'fill_pct']
    numeric_features = ['max_vol', 'fill_pct', 'tank volume', 'lat', 'lon', 'station_dist',
                        'temperature', 'dwpt', 'rhum', 'wspd', 'distance_to_nearest_port',
                        'distance_to_major_city', 'distance_to_coast', 'days_since_eia_report',
                        'wti_price', 'brent_price', 'wti_brent_spread']

    analyzer = CorrelationAnalyzer(
        data_path='full_data.csv',
        target_features=target_features,
        numeric_features=numeric_features,
        output_dir='/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/corr'
    )
    analyzer.run_analysis()