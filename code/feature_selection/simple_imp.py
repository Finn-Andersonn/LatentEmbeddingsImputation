# -----------------------------------------
# Ran simple imputation methods for corr.py
#
# Also used later for framing more advanced methods
# Also will emphasize KNN here, before its used in latent embeddings
# -----------------------------------------

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def evaluate_imputation(original_series, imputed_series):
    """
    Description:
        Evaluating the RMSE & MAE only on the known (non-NaN) data points.

    Args:
        original_series: pandas dataframe
        imputed_series: pandas dataframe
    """
    mask = ~original_series.isna()  # this is true where original_series is not NaN
    rmse = np.sqrt(mean_squared_error(original_series[mask], imputed_series[mask]))
    mae = mean_absolute_error(original_series[mask], imputed_series[mask])
    return rmse, mae

# original dataset
data_path = "/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/missingness/complete_panel_with_features_updated.csv"
df = pd.read_csv(data_path)

if 'season' in df.columns and 'season_numeric' not in df.columns:
    df['season_numeric'] = df['season'].map({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}).astype(float)


numeric_features = [
        "fill_pct", "tank volume", "max_vol", "temperature_avg", "temperature_min",
        "temperature_max", "precipitation", "wind_speed", "wind_dir",
        "pressure",
        "distance_to_nearest_port", "distance_to_nearest_tank_id",
        "wti_price", "brent_price", "wti_brent_spread", "season_numeric"
    ]

# subset df to numeric features
df_numeric = df[numeric_features].copy()

# we artificially introduce missingness to evaluate imputation
print("\nIntroducing missingness to evaluate imputation methods...")
np.random.seed(42)
mask = np.random.rand(len(df_numeric)) < 0.1 #  (10% missing) 
df_missing = df_numeric.copy()
for col in df_numeric.columns:
    df_missing.loc[mask, col] = np.nan

# 1: Median Imputation
print("\nImputing missing values with median...")
df_median = df_missing.copy()
for col in df_median.columns:
    df_median[col] = df_median[col].fillna(df_median[col].median())

# 2: KNN Imputation (KEY) <- used in latent embedding later
print("\nImputing missing values with KNN...")
df_knn = df_missing.copy()
imputer_knn = KNNImputer(n_neighbors=5, weights='distance') # tweaked neighbours until 5
knn_imputed = imputer_knn.fit_transform(df_knn)
df_knn = pd.DataFrame(knn_imputed, columns=df_knn.columns, index=df_knn.index)

# 3: Iterative Imputation (MICE)
print("\nImputing missing values with Iterative Imputation (MICE)...")
df_iter = df_missing.copy()
imputer_iter = IterativeImputer(random_state=42)
iter_imputed = imputer_iter.fit_transform(df_iter)
df_iter = pd.DataFrame(iter_imputed, columns=df_iter.columns, index=df_iter.index)

# evaluate
print("\nEvaluating imputation methods...")
results = {}
for col in numeric_features:
    results[col] = {
        'Median': evaluate_imputation(df_numeric[col], df_median[col]), # we give the original and imputed dataframe for each method
        'KNN': evaluate_imputation(df_numeric[col], df_knn[col]),
        'Iterative': evaluate_imputation(df_numeric[col], df_iter[col])
    }

print("\nImputation Evaluation Results (RMSE, MAE):")
for col in numeric_features:
    print(f"\nFeature: {col}")
    for method in results[col]:
        rmse, mae = results[col][method]
        print(f"  {method}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# we'll set the best method (average RMSE across features) as the highest average RMSE
avg_rmse = {
    'Median': np.mean([results[col]['Median'][0] for col in numeric_features]),
    'KNN': np.mean([results[col]['KNN'][0] for col in numeric_features]),
    'Iterative': np.mean([results[col]['Iterative'][0] for col in numeric_features])
}

print("\nAverage RMSE across features:")
for method, rmse in avg_rmse.items():
    print(f"  {method}: {rmse:.4f}")

best_method = min(avg_rmse, key=avg_rmse.get)
print(f"\nBest imputation method for VIF calculation: {best_method}")