# ---------------------------------------
# Tools used across Autoencoder + SAITS methods
#   -> setting a baseline with these advanced methods
#   -> measuring memory usage, time, 
#
# ---------------------------------------

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import psutil
import os
import matplotlib.pyplot as plt

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Memory in MB

def prep():
    # Train-test split preserving temporal order
    def train_test_split_temporal(df, test_size=0.2):
        """
        Split the data into train and test sets, preserving temporal order.
        Assumes df is sorted by time.
        """
        n_rows = len(df)
        train_size = int(n_rows * (1 - test_size))
        train_idx = np.arange(0, train_size)
        test_idx = np.arange(train_size, n_rows)
        return train_idx, test_idx

    data_path = "/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/missingness/complete_panel_with_features_updated.csv"
    df = pd.read_csv(data_path)

    # sort by imaging_time to ensure temporal order
    df['imaging_time'] = pd.to_datetime(df['imaging_time'], errors='coerce')
    df = df.sort_values(['tank_id', 'imaging_time']).reset_index(drop=True)  # Sort by tank_id and time

    if 'season' in df.columns and 'season_numeric' not in df.columns:
        df['season_numeric'] = df['season'].map({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}).astype(float)

    # features (before lags and rolling)
    base_features = [
        'fill_pct', 'distance_to_nearest_tank_id', 'wind_speed', 'temperature_avg',
        'temperature_min', 'wti_brent_spread', 'wind_dir', 'distance_to_nearest_port',
        'wti_price', 'season_numeric'
    ]

    # base features exist and are numeric
    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features {missing_features} not found in dataset")
    for col in base_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # lagged and rolling features
    max_lags = 3
    static_features = ['distance_to_nearest_port', 'distance_to_nearest_tank_id']
    dynamic_features = [f for f in base_features if f not in static_features]

    # lagged features
    for col in base_features:
        for lag in range(1, max_lags + 1):
            df[f'{col}_lag{lag}'] = df.groupby('tank_id')[col].shift(lag)

    # rolling statistics for dynamic features
    for col in dynamic_features:
        df[f'{col}_roll_mean'] = (df.groupby('tank_id')[col]
                                .rolling(window=7, min_periods=1)
                                .mean()
                                .reset_index(level=0, drop=True))
        df[f'{col}_roll_std'] = (df.groupby('tank_id')[col]
                                .rolling(window=7, min_periods=1)
                                .std()
                                .reset_index(level=0, drop=True))

    # Final feature set
    final_features = [
        'distance_to_nearest_port_lag3', 'wind_speed_roll_mean', 'fill_pct', 'fill_pct_lag1',
        'fill_pct_lag2', 'fill_pct_lag3', 'fill_pct_roll_mean', 'fill_pct_roll_std',
        'distance_to_nearest_tank_id_lag3', 'temperature_avg', 'temperature_min_roll_mean',
        'wti_brent_spread_roll_mean', 'wind_dir_roll_mean', 'wti_price', 'season_numeric'
    ]

    df_selected = df[final_features].copy()

    train_idx, test_idx = train_test_split_temporal(df_selected, test_size=0.3)  # 70/30 split
    df_train = df_selected.iloc[train_idx].copy()
    df_test = df_selected.iloc[test_idx].copy()

    # missingness in train and test sets
    print(f"Training set: {len(df_train)} rows, {df_train['fill_pct'].notna().sum()} non-missing fill_pct values")
    print(f"Test set: {len(df_test)} rows, {df_test['fill_pct'].notna().sum()} non-missing fill_pct values")

    # Normalize the data (fit on training data only)
    df_mean = df_train.mean()
    df_std = df_train.std()
    df_std = df_std.replace(0, 1e-6)  # Avoid division by zero
    df_train_normalized = (df_train - df_mean) / df_std
    df_test_normalized = (df_test - df_mean) / df_std
    df_train_normalized = df_train_normalized.fillna(0)  # Fill NaNs with 0 for training
    df_test_normalized = df_test_normalized.fillna(0)    # Fill NaNs with 0 for testing

    # Normalize the entire dataset for final imputation
    df_full_normalized = (df_selected - df_mean) / df_std
    df_full_normalized = df_full_normalized.fillna(0)

    return final_features, df_full_normalized, df_train_normalized, df_test_normalized, df_train, df_test, df_std, df_mean, df

def evaluate_imputation(original_series, imputed_series, mask):
    '''
    Description:
        Evaluates the imputation performance on the specified mask.
        Returns RMSE, MAE, MAPE, and R².
    Args:
        original_series: pandas dataframe
        imputed_series: pandas dataframe
        mask: cover certain of values
    Returns:
        RMSE, MAE, MAPE, R2
    '''
    if not mask.any():  # If no values to evaluate, return None
        return None, None, None, None
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(original_series[mask], imputed_series[mask]))
    # Compute MAE
    mae = mean_absolute_error(original_series[mask], imputed_series[mask])
    # Compute MAPE (Mean Absolute Percentage Error)
    abs_error = np.abs(original_series[mask] - imputed_series[mask])
    mape = np.mean(abs_error / (original_series[mask] + 1e-6)) * 100  # Avoid division by zero
    # Compute R²
    r2 = r2_score(original_series[mask], imputed_series[mask])
    return rmse, mae, mape, r2

# walk-forward CV split for the training set
def walk_forward_split(df, n_splits=5):
    '''
    Description:
        Split the data into walk-forward train/validation sets.
        Preserves temporal order -> we sorted df by time before
    Args:
        df: pandas dataframe
        n_splits: number of folds
    Returns:
        splits: the areas of splits
    '''
    n_rows = len(df)
    step_size = n_rows // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = i * step_size
        val_end = (i + 1) * step_size
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        splits.append((train_idx, val_idx))
    return splits

# track per-epoch losses
def track_losses(model_name, fold, epoch, loss, loss_dict):
    '''
    For plotting
    '''
    if isinstance(fold, int):
        key = f"{model_name}_Fold_{fold+1}"
    else:
        key = f"{model_name}_{fold}"  # e.g. "Autoencoder_Final"
    
    if key not in loss_dict:
        loss_dict[key] = []
    loss_dict[key].append(loss)


# below is to set the common graph style for the paper
def set_common_plot_style(ax, title):
    ax.set_facecolor('#f5f5f5')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=12, colors='gray')
    ax.grid(True, linestyle='--', color='gray', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('gray')
        ax.spines[spine].set_linewidth(0.5)

def plot_epoch_losses(loss_dict, model_name, save_path="epoch_losses"):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for key, losses in loss_dict.items():
        if model_name in key:
            ax.plot(range(1, len(losses) + 1), losses, label=key, alpha=0.8)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    set_common_plot_style(ax, f"{model_name} Per-Epoch Loss Across Folds")
    
    ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5),
              frameon=True, edgecolor='black', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{save_path}_{model_name.lower()}.png")
    plt.close()


def plot_cv_performance(cv_metrics, model_name, metric_name="RMSE", save_path="cv_performance"):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    folds = list(range(1, len(cv_metrics) + 1))
    ax.plot(folds, cv_metrics, marker='o', linestyle='-', color='b', alpha=0.8, label=metric_name)

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    set_common_plot_style(ax, f"{model_name} {metric_name} Across CV Folds")
    
    ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5),
              frameon=True, edgecolor='black', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{save_path}_{model_name.lower()}_{metric_name.lower()}.png")
    plt.close()


def plot_imputation_comparison(original_series, imputed_series, mask, model_name, num_points=100, save_path="imputation_comparison"):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Filter for masked points
    original_masked = original_series[mask]
    imputed_masked = imputed_series[mask]
    indices = np.arange(len(original_masked))
    
    if len(indices) > num_points:
        step = len(indices) // num_points
        indices = indices[::step]
        original_masked = original_masked[::step]
        imputed_masked = imputed_masked[::step]

    ax.plot(indices, original_masked, label="Original", marker='o', linestyle='-', color='blue', alpha=0.6)
    ax.plot(indices, imputed_masked, label="Imputed", marker='x', linestyle='--', color='red', alpha=0.6)

    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("fill_pct", fontsize=12)
    set_common_plot_style(ax, f"{model_name} Imputation: Original vs. Imputed fill_pct")

    ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5),
              frameon=True, edgecolor='black', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{save_path}_{model_name.lower()}.png")
    plt.close()
