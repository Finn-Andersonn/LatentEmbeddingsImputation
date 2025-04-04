# ---------------------------------------------
# First advanced model
#   -> Autoencoder
#   -> Dimensionality reduction
#   -> Loss on reconstruction
#   -> Offers different, yet comparable, baseline for SAITS
# ---------------------------------------------


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from imp_prep import evaluate_imputation, track_losses, plot_epoch_losses, plot_cv_performance, plot_imputation_comparison, get_memory_usage

# 4. Autoencoder Imputation
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder_cv(encoding_dims, splits, n_splits, df_train_normalized, df_train, input_dim, df_std, df_mean, fill_pct_idx):
    '''
    Description:
        Perform walk-forward CV and hyperparameter tuning for the Autoencoder.
    
    Args:
        encoding_dims       : dimensions hyperparameter
        splits              : splits for CV
        n_splits            : number of splits for CV
        df_train_normalized : df regularized
        df_train            : original df
        input_dim           : input dimensions hyperparameter
        df_std              : used for reg
        df_mean             : used for reg
        fill_pct_idx        : fill_pct index

    Returns:
        Returns the best encoding_dim, computational metrics, and loss dictionary.
    '''
    comp_metrics_ae = []
    loss_dict_ae = {}
    best_rmse_ae = float('inf')
    best_encoding_dim = None

    for encoding_dim in encoding_dims:
        print(f"\nTuning Autoencoder with encoding_dim={encoding_dim}")
        rmse_scores = []
        for fold, (train_idx_fold, val_idx_fold) in enumerate(splits):
            print(f"Fold {fold + 1}/{n_splits}")
            # prep + standardize
            train_data = df_train_normalized.iloc[train_idx_fold].values.astype(np.float32)
            val_data = df_train_normalized.iloc[val_idx_fold].values.astype(np.float32)
            train_tensor = torch.from_numpy(train_data)
            val_tensor = torch.from_numpy(val_data)
            mask_train = ~df_train.iloc[train_idx_fold].isna().values
            mask_val = ~df_train.iloc[val_idx_fold].isna().values

            # train
            model_ae = SimpleAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model_ae.parameters(), lr=1e-3)
            epochs = 100

            # training time and memory measurement
            start_time = time.time()
            start_mem = get_memory_usage()
            for epoch in range(epochs):
                model_ae.train()
                optimizer.zero_grad()
                outputs = model_ae(train_tensor)
                loss = criterion(outputs[mask_train], train_tensor[mask_train])
                loss.backward()
                optimizer.step()
                # loss
                track_losses("Autoencoder", fold, epoch, loss.item(), loss_dict_ae)
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            train_time = time.time() - start_time
            peak_mem = get_memory_usage()
            train_mem = peak_mem - start_mem

            #eval on validation set
            start_time = time.time()
            start_mem = get_memory_usage()
            model_ae.eval()
            with torch.no_grad():
                val_outputs = model_ae(val_tensor).numpy()
            inference_time = time.time() - start_time
            peak_mem = get_memory_usage()
            inference_mem = peak_mem - start_mem

            # denormalize
            val_outputs = (val_outputs * df_std.values) + df_mean.values
            val_original = df_train.iloc[val_idx_fold]["fill_pct"]
            val_imputed = pd.Series(val_outputs[:, fill_pct_idx], index=val_original.index)
            rmse, mae, mape, r2 = evaluate_imputation(val_original, val_imputed, ~val_original.isna())
            if rmse is None:
                print(f"Fold {fold + 1} skipped: No non-missing fill_pct values in validation set.")
                continue
            rmse_scores.append(rmse)
            print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")

            # record for this fold
            comp_metrics_ae.append({
                'Hyperparams': f"encoding_dim={encoding_dim}",
                'Fold': fold + 1,
                'Train Time (s)': train_time,
                'Train Memory (MB)': train_mem,
                'Inference Time (s)': inference_time,
                'Inference Memory (MB)': inference_mem,
                'Inference Time Simulated (s)': None,
                'Inference Memory Simulated (MB)': None,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            })

        if not rmse_scores:  # Fallback if all folds were skipped
            print(f"No valid folds for encoding_dim={encoding_dim}. Skipping.")
            continue
        avg_rmse = np.mean(rmse_scores)
        print(f"Average RMSE for encoding_dim={encoding_dim}: {avg_rmse:.4f} ± {np.std(rmse_scores):.4f}")
        if avg_rmse < best_rmse_ae:
            best_rmse_ae = avg_rmse
            best_encoding_dim = encoding_dim

    if best_encoding_dim is None:
        print("No valid folds found for Autoencoder tuning. Using default encoding_dim=8.")
        best_encoding_dim = 8

    print(f"\nBest Autoencoder encoding_dim: {best_encoding_dim}, RMSE: {best_rmse_ae:.4f}")

    # CV performance
    rmse_scores_ae = [metric['RMSE'] for metric in comp_metrics_ae if metric['RMSE'] is not None]
    if rmse_scores_ae:
        plot_cv_performance(rmse_scores_ae, "Autoencoder", metric_name="RMSE")

    # epoch losses
    plot_epoch_losses(loss_dict_ae, "Autoencoder")

    return best_encoding_dim, comp_metrics_ae, loss_dict_ae

def run_autoencoder_final(best_encoding_dim, df_train_normalized, df_train, df_test_normalized, df_test, df_full_normalized, df, input_dim, df_std, df_mean, fill_pct_idx, comp_metrics_ae, loss_dict_ae):
    """
    Description:
        Same as above, but training the final Autoencoder model on the entire training set, evaluate on the test set and full dataset,
        and generate plots.
    """
    train_tensor = torch.from_numpy(df_train_normalized.values.astype(np.float32))
    mask_ae = ~df_train.isna().values
    model_ae = SimpleAutoencoder(input_dim=input_dim, encoding_dim=best_encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_ae.parameters(), lr=1e-3)
    epochs = 100

    # training time and memory again
    start_time = time.time()
    start_mem = get_memory_usage()
    for epoch in range(epochs):
        model_ae.train()
        optimizer.zero_grad()
        outputs = model_ae(train_tensor)
        loss = criterion(outputs[mask_ae], train_tensor[mask_ae])
        loss.backward()
        optimizer.step()
        # loss
        track_losses("Autoencoder", "Final", epoch, loss.item(), loss_dict_ae)
        if (epoch + 1) % 20 == 0:
            print(f"Final Autoencoder Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    final_train_time = time.time() - start_time
    peak_mem = get_memory_usage()
    final_train_mem = peak_mem - start_mem

    # simulate missingness in the test set for evaluation
    df_ae_test = df_test.copy()
    non_missing_mask = df_ae_test["fill_pct"].notna()
    if not non_missing_mask.any():
        print("[Autoencoder] No non-missing fill_pct values in the test set to evaluate.")
    else:
        np.random.seed(42)  # For reproducibility
        mask_to_impute = non_missing_mask & (np.random.rand(len(df_ae_test)) < 0.2)
        print(f"Simulating missingness: Masking {mask_to_impute.sum()} non-missing fill_pct values for evaluation.")

        df_ae_test_simulated = df_ae_test.copy()
        df_ae_test_simulated.loc[mask_to_impute, "fill_pct"] = np.nan

        df_ae_test_simulated_normalized = (df_ae_test_simulated - df_mean) / df_std
        df_ae_test_simulated_normalized = df_ae_test_simulated_normalized.fillna(0)

        # Measure inference time and memory for the test set
        start_time = time.time()
        start_mem = get_memory_usage()
        test_tensor = torch.from_numpy(df_ae_test_simulated_normalized.values.astype(np.float32))
        model_ae.eval()
        with torch.no_grad():
            reconstructed_ae = model_ae(test_tensor).numpy()
        inference_time = time.time() - start_time
        peak_mem = get_memory_usage()
        inference_mem = peak_mem - start_mem

        reconstructed_ae = (reconstructed_ae * df_std.values) + df_mean.values

        fill_pct_ae = df_ae_test_simulated["fill_pct"].copy()
        fill_pct_mask = df_ae_test_simulated["fill_pct"].isna()
        fill_pct_ae[fill_pct_mask] = reconstructed_ae[fill_pct_mask, fill_pct_idx]
        df_ae_test_simulated["fill_pct_ae"] = fill_pct_ae

        rmse_ae, mae_ae, mape_ae, r2_ae = evaluate_imputation(df_ae_test["fill_pct"], df_ae_test_simulated["fill_pct_ae"], mask_to_impute)
        if rmse_ae is None or mae_ae is None:
            print("[Autoencoder] Cannot evaluate: No artificially masked fill_pct values in test set.")
        else:
            print(f"[Autoencoder] Test RMSE: {rmse_ae:.4f}, Test MAE: {mae_ae:.4f}, Test MAPE: {mape_ae:.4f}, Test R2: {r2_ae:.4f}")

            # imputation comparison for the test set
            plot_imputation_comparison(df_ae_test["fill_pct"], df_ae_test_simulated["fill_pct_ae"], mask_to_impute, "Autoencoder_Test")

            # computational metrics for the test set
            comp_metrics_ae.append({
                'Hyperparams': f"encoding_dim={best_encoding_dim}",
                'Fold': 'Test',
                'Train Time (s)': final_train_time,
                'Train Memory (MB)': final_train_mem,
                'Inference Time (s)': inference_time,
                'Inference Memory (MB)': inference_mem,
                'Inference Time Simulated (s)': None,
                'Inference Memory Simulated (MB)': None,
                'RMSE': rmse_ae,
                'MAE': mae_ae,
                'MAPE': mape_ae,
                'R2': r2_ae
            })

    # Simulate missingness on the entire dataset for evaluation
    df_ae_full = df.copy()
    non_missing_mask_full = df_ae_full["fill_pct"].notna()
    if not non_missing_mask_full.any():
        print("[Autoencoder] No non-missing fill_pct values in the full dataset to evaluate.")
    else:
        np.random.seed(42)  # For reproducibility
        mask_to_impute_full = non_missing_mask_full & (np.random.rand(len(df_ae_full)) < 0.2)
        print(f"Simulating missingness on full dataset: Masking {mask_to_impute_full.sum()} non-missing fill_pct values for evaluation.")

        df_ae_full_simulated = df_ae_full.copy()
        df_ae_full_simulated.loc[mask_to_impute_full, "fill_pct"] = np.nan

        # **Fix:** Restrict normalization to the numeric columns used during training
        df_ae_full_simulated_numeric = df_ae_full_simulated[df_mean.index]
        df_ae_full_simulated_normalized = (df_ae_full_simulated_numeric - df_mean) / df_std
        df_ae_full_simulated_normalized = df_ae_full_simulated_normalized.fillna(0)

        # Measure inference time and memory for the full dataset
        start_time = time.time()
        start_mem = get_memory_usage()
        full_tensor = torch.from_numpy(df_ae_full_simulated_normalized.values.astype(np.float32))
        model_ae.eval()
        with torch.no_grad():
            reconstructed_ae_full = model_ae(full_tensor).numpy()
        inference_time_full = time.time() - start_time
        peak_mem = get_memory_usage()
        inference_mem_full = peak_mem - start_mem

        reconstructed_ae_full = (reconstructed_ae_full * df_std.values) + df_mean.values

        fill_pct_ae_full = df_ae_full_simulated["fill_pct"].copy()
        fill_pct_mask_full = df_ae_full_simulated["fill_pct"].isna()
        fill_pct_ae_full[fill_pct_mask_full] = reconstructed_ae_full[fill_pct_mask_full, fill_pct_idx]
        df_ae_full_simulated["fill_pct_ae"] = fill_pct_ae_full

        rmse_ae_full, mae_ae_full, mape_ae_full, r2_ae_full = evaluate_imputation(df_ae_full["fill_pct"], df_ae_full_simulated["fill_pct_ae"], mask_to_impute_full)
        if rmse_ae_full is None or mae_ae_full is None:
            print("[Autoencoder] Cannot evaluate: No artificially masked fill_pct values in full dataset.")
        else:
            print(f"[Autoencoder] Full Dataset RMSE: {rmse_ae_full:.4f}, Full Dataset MAE: {mae_ae_full:.4f}, Full Dataset MAPE: {mape_ae_full:.4f}, Full Dataset R2: {r2_ae_full:.4f}")

            # imputation comparison for the full dataset
            plot_imputation_comparison(df_ae_full["fill_pct"], df_ae_full_simulated["fill_pct_ae"], mask_to_impute_full, "Autoencoder_Full")

            # computational metrics for the full dataset
            comp_metrics_ae.append({
                'Hyperparams': f"encoding_dim={best_encoding_dim}",
                'Fold': 'Final',
                'Train Time (s)': final_train_time,
                'Train Memory (MB)': final_train_mem,
                'Inference Time (s)': inference_time_full,
                'Inference Memory (MB)': inference_mem_full,
                'Inference Time Simulated (s)': None,
                'Inference Memory Simulated (MB)': None,
                'RMSE': rmse_ae_full,
                'MAE': mae_ae_full,
                'MAPE': mape_ae_full,
                'R2': r2_ae_full
            })

    # epoch losses for the final model
    plot_epoch_losses(loss_dict_ae, "Autoencoder")

    # computational metrics to a CSV
    comp_metrics_ae_df = pd.DataFrame(comp_metrics_ae)
    comp_metrics_ae_df.to_csv("autoencoder_comp_metrics.csv", index=False)
    print("Saved computational metrics to 'autoencoder_comp_metrics.csv'")

    return model_ae, comp_metrics_ae, loss_dict_ae