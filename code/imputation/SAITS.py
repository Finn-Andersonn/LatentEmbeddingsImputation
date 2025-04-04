# --------------------------------------
# First advanced model:
#   -> Self-attentions transformer
#   -> Suspected advantages:
#   -> Potential disadvantages:
#       ~ overfitting (even with excessive regularization)
# --------------------------------------


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from imp_prep import evaluate_imputation, track_losses, plot_epoch_losses, plot_cv_performance, plot_imputation_comparison, get_memory_usage

# 5. Self-Attention Imputation (SAITS)
class SAITSImputer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(SAITSImputer, self).__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x_embed = self.linear_in(x)
        encoded = self.transformer_encoder(x_embed)
        out = self.linear_out(encoded)
        return out

def train_saits_cv(d_model_options, nhead_options, num_layers_options, n_splits, splits, df_train_normalized, df_train, input_dim, df_std, df_mean, fill_pct_idx):
    """
    Description:
        Same split as in imp_prep, but includes training for SAITS
        Perform walk-forward CV and hyperparameter tuning for SAITS.

    Args:
        d_model_options     : dimensions hyperparameter
        nhead_options       : heads hyperparameter
        num_layers_options  : layers hyperparameter
        n_splits            : selected for walk-forward CV
        plits               : from walk-forward CV
        df_train_normalized : adjusted trainin set
        df_train            : original training set
        input_dim           : len(final_features)
        df_std              : std to regularize
        df_mean             : mean to regularize
        fill_pct_idx        : fill_pct index
    
    Returns:
        best parameters, computational metrics, and loss dictionary.
    """
    comp_metrics_saits = []
    loss_dict_saits = {}
    best_rmse_saits = float('inf')
    best_saits_params = None

    for d_model, nhead in zip(d_model_options, nhead_options):
        for num_layers in num_layers_options:
            print(f"\nTuning SAITS with d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
            rmse_scores = []
            for fold, (train_idx_fold, val_idx_fold) in enumerate(splits):
                print(f"Fold {fold + 1}/{n_splits}")
                # Prepare data
                train_data = df_train_normalized.iloc[train_idx_fold].values.astype(np.float32)
                val_data = df_train_normalized.iloc[val_idx_fold].values.astype(np.float32)
                train_tensor = torch.from_numpy(train_data).unsqueeze(1)
                val_tensor = torch.from_numpy(val_data).unsqueeze(1)
                mask_train = ~df_train.iloc[train_idx_fold].isna().values
                mask_val = ~df_train.iloc[val_idx_fold].isna().values

                # Train the SAITS model
                model_saits = SAITSImputer(
                    input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=0.1
                )
                criterion_saits = nn.MSELoss()
                optimizer_saits = optim.Adam(model_saits.parameters(), lr=1e-3, weight_decay=1e-5)
                scheduler = optim.lr_scheduler.StepLR(optimizer_saits, step_size=20, gamma=0.5)
                epochs_saits = 100

                # Measure training time and memory
                start_time = time.time()
                start_mem = get_memory_usage()
                for epoch in range(epochs_saits):
                    model_saits.train()
                    optimizer_saits.zero_grad()
                    outputs = model_saits(train_tensor)
                    loss = criterion_saits(outputs.squeeze(1)[mask_train], train_tensor.squeeze(1)[mask_train])
                    loss.backward()
                    optimizer_saits.step()
                    scheduler.step()
                    # Track loss
                    track_losses("SAITS", fold, epoch, loss.item(), loss_dict_saits)
                    if (epoch + 1) % 20 == 0:
                        print(f"Epoch {epoch + 1}/{epochs_saits}, Loss: {loss.item():.4f}")
                train_time = time.time() - start_time
                peak_mem = get_memory_usage()
                train_mem = peak_mem - start_mem

                # Evaluate on validation set
                start_time = time.time()
                start_mem = get_memory_usage()
                model_saits.eval()
                with torch.no_grad():
                    val_outputs = model_saits(val_tensor).squeeze(1).numpy()
                inference_time = time.time() - start_time
                peak_mem = get_memory_usage()
                inference_mem = peak_mem - start_mem

                # Denormalize
                val_outputs = (val_outputs * df_std.values) + df_mean.values
                val_original = df_train.iloc[val_idx_fold]["fill_pct"]
                val_imputed = pd.Series(val_outputs[:, fill_pct_idx], index=val_original.index)
                rmse, mae, mape, r2 = evaluate_imputation(val_original, val_imputed, ~val_original.isna())
                if rmse is None:
                    print(f"Fold {fold + 1} skipped: No non-missing fill_pct values in validation set.")
                    continue
                rmse_scores.append(rmse)
                print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")

                # Record computational metrics for this fold
                comp_metrics_saits.append({
                    'Hyperparams': f"d_model={d_model}, nhead={nhead}, num_layers={num_layers}",
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

            if not rmse_scores:
                print(f"No valid folds for d_model={d_model}, nhead={nhead}, num_layers={num_layers}. Skipping.")
                continue
            avg_rmse = np.mean(rmse_scores)
            print(f"Average RMSE for d_model={d_model}, nhead={nhead}, num_layers={num_layers}: {avg_rmse:.4f} ± {np.std(rmse_scores):.4f}")
            if avg_rmse < best_rmse_saits:
                best_rmse_saits = avg_rmse
                best_saits_params = (d_model, nhead, num_layers)

    if best_saits_params is None:
        print("No valid folds found for SAITS tuning. Using default parameters: d_model=16, nhead=4, num_layers=2.")
        best_saits_params = (16, 4, 2)

    print(f"\nBest SAITS params: d_model={best_saits_params[0]}, nhead={best_saits_params[1]}, num_layers={best_saits_params[2]}, RMSE: {best_rmse_saits:.4f}")

    # CV performance
    rmse_scores_saits = [metric['RMSE'] for metric in comp_metrics_saits if metric['RMSE'] is not None]
    if rmse_scores_saits:
        plot_cv_performance(rmse_scores_saits, "SAITS", metric_name="RMSE")

    # epoch losses
    plot_epoch_losses(loss_dict_saits, "SAITS")

    return best_saits_params, comp_metrics_saits, loss_dict_saits

def run_saits_final(best_saits_params, df_train_normalized, df_train, df_test_normalized, df_test, df_full_normalized, df, input_dim, df_std, df_mean, fill_pct_idx, comp_metrics_saits, loss_dict_saits):
    """
    Description:
        Same as the above, but now training the final SAITS (i.e. after ideal hyperparameters) model on the entire training set, evaluate 
        on the test set and full dataset, and generate plots.
    """
    train_saits_tensor = torch.from_numpy(df_train_normalized.values.astype(np.float32)).unsqueeze(1)
    mask_saits = ~df_train.isna().values
    model_saits = SAITSImputer(
        input_dim=input_dim, d_model=best_saits_params[0], nhead=best_saits_params[1], num_layers=best_saits_params[2], dropout=0.1
    )
    criterion_saits = nn.MSELoss()
    optimizer_saits = optim.Adam(model_saits.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer_saits, step_size=20, gamma=0.5)
    epochs_saits = 100

    # training time and memory measurement
    start_time = time.time()
    start_mem = get_memory_usage()
    for epoch in range(epochs_saits):
        model_saits.train()
        optimizer_saits.zero_grad()
        outputs = model_saits(train_saits_tensor)
        loss = criterion_saits(outputs.squeeze(1)[mask_saits], train_saits_tensor.squeeze(1)[mask_saits])
        loss.backward()
        optimizer_saits.step()
        scheduler.step()
        # loss for the final model
        track_losses("SAITS", "Final", epoch, loss.item(), loss_dict_saits)
        if (epoch + 1) % 20 == 0:
            print(f"Final SAITS Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    final_train_time = time.time() - start_time
    peak_mem = get_memory_usage()
    final_train_mem = peak_mem - start_mem

    # simulate missingness in the test set for SAITS evaluation (just like in simple_imp.py)
    df_saits_test = df_test.copy()
    non_missing_mask = df_saits_test["fill_pct"].notna()
    if not non_missing_mask.any():
        print("[SAITS] No non-missing fill_pct values in the test set to evaluate.")
    else:
        np.random.seed(42)  # for reproducibility
        mask_to_impute = non_missing_mask & (np.random.rand(len(df_saits_test)) < 0.2)
        print(f"Simulating missingness: Masking {mask_to_impute.sum()} non-missing fill_pct values for evaluation.")

        df_saits_test_simulated = df_saits_test.copy()
        df_saits_test_simulated.loc[mask_to_impute, "fill_pct"] = np.nan

        df_saits_test_simulated_normalized = (df_saits_test_simulated - df_mean) / df_std
        df_saits_test_simulated_normalized = df_saits_test_simulated_normalized.fillna(0)

        # inference time and memory for the test set measurement
        start_time = time.time()
        start_mem = get_memory_usage()
        test_saits_tensor = torch.from_numpy(df_saits_test_simulated_normalized.values.astype(np.float32)).unsqueeze(1)
        model_saits.eval()
        with torch.no_grad():
            reconstructed_saits = model_saits(test_saits_tensor).squeeze(1).numpy()
        inference_time = time.time() - start_time
        peak_mem = get_memory_usage()
        inference_mem = peak_mem - start_mem

        # put values back
        reconstructed_saits = (reconstructed_saits * df_std.values) + df_mean.values

        fill_pct_saits = df_saits_test_simulated["fill_pct"].copy()
        fill_pct_mask = df_saits_test_simulated["fill_pct"].isna()
        fill_pct_saits[fill_pct_mask] = reconstructed_saits[fill_pct_mask, fill_pct_idx]
        df_saits_test_simulated["fill_pct_saits"] = fill_pct_saits

        rmse_saits, mae_saits, mape_saits, r2_saits = evaluate_imputation(df_saits_test["fill_pct"], df_saits_test_simulated["fill_pct_saits"], mask_to_impute)
        if rmse_saits is None or mae_saits is None:
            print("[SAITS] Cannot evaluate: No artificially masked fill_pct values in test set.")
        else:
            print(f"[SAITS] Test RMSE: {rmse_saits:.4f}, Test MAE: {mae_saits:.4f}, Test MAPE: {mape_saits:.4f}, Test R2: {r2_saits:.4f}")

            # imputation comparison for the test set
            plot_imputation_comparison(df_saits_test["fill_pct"], df_saits_test_simulated["fill_pct_saits"], mask_to_impute, "SAITS_Test")

            # computational metrics for the test set
            comp_metrics_saits.append({
                'Hyperparams': f"d_model={best_saits_params[0]}, nhead={best_saits_params[1]}, num_layers={best_saits_params[2]}",
                'Fold': 'Test',
                'Train Time (s)': final_train_time,
                'Train Memory (MB)': final_train_mem,
                'Inference Time (s)': inference_time,
                'Inference Memory (MB)': inference_mem,
                'Inference Time Simulated (s)': None,
                'Inference Memory Simulated (MB)': None,
                'RMSE': rmse_saits,
                'MAE': mae_saits,
                'MAPE': mape_saits,
                'R2': r2_saits
            })

    # simulate missingness again on the entire dataset for evaluation
    df_saits_full = df.copy()
    non_missing_mask_full = df_saits_full["fill_pct"].notna()
    if not non_missing_mask_full.any():
        print("[SAITS] No non-missing fill_pct values in the full dataset to evaluate.")
    else:
        np.random.seed(42)  # again for reproducibility
        mask_to_impute_full = non_missing_mask_full & (np.random.rand(len(df_saits_full)) < 0.2)
        print(f"Simulating missingness on full dataset: Masking {mask_to_impute_full.sum()} non-missing fill_pct values for evaluation.")

        df_saits_full_simulated = df_saits_full.copy()
        df_saits_full_simulated.loc[mask_to_impute_full, "fill_pct"] = np.nan

        df_saits_full_simulated_normalized = (df_saits_full_simulated - df_mean) / df_std
        df_saits_full_simulated_normalized = df_saits_full_simulated_normalized.fillna(0)

        # inference time and memory for the full dataset again
        start_time = time.time()
        start_mem = get_memory_usage()
        full_saits_tensor = torch.from_numpy(df_saits_full_simulated_normalized.values.astype(np.float32)).unsqueeze(1)
        model_saits.eval()
        with torch.no_grad():
            reconstructed_saits_full = model_saits(full_saits_tensor).squeeze(1).numpy()
        inference_time_full = time.time() - start_time
        peak_mem = get_memory_usage()
        inference_mem_full = peak_mem - start_mem

        reconstructed_saits_full = (reconstructed_saits_full * df_std.values) + df_mean.values

        fill_pct_saits_full = df_saits_full_simulated["fill_pct"].copy()
        fill_pct_mask_full = df_saits_full_simulated["fill_pct"].isna()
        fill_pct_saits_full[fill_pct_mask_full] = reconstructed_saits_full[fill_pct_mask_full, fill_pct_idx]
        df_saits_full_simulated["fill_pct_saits"] = fill_pct_saits_full

        rmse_saits_full, mae_saits_full, mape_saits_full, r2_saits_full = evaluate_imputation(df_saits_full["fill_pct"], df_saits_full_simulated["fill_pct_saits"], mask_to_impute_full)
        if rmse_saits_full is None or mae_saits_full is None:
            print("[SAITS] Cannot evaluate: No artificially masked fill_pct values in full dataset.")
        else:
            print(f"[SAITS] Full Dataset RMSE: {rmse_saits_full:.4f}, Full Dataset MAE: {mae_saits_full:.4f}, Full Dataset MAPE: {mape_saits_full:.4f}, Full Dataset R2: {r2_saits_full:.4f}")

            # imputation comparison for the full dataset
            plot_imputation_comparison(df_saits_full["fill_pct"], df_saits_full_simulated["fill_pct_saits"], mask_to_impute_full, "SAITS_Full")

            # computational metrics for the full dataset
            comp_metrics_saits.append({
                'Hyperparams': f"d_model={best_saits_params[0]}, nhead={best_saits_params[1]}, num_layers={best_saits_params[2]}",
                'Fold': 'Final',
                'Train Time (s)': final_train_time,
                'Train Memory (MB)': final_train_mem,
                'Inference Time (s)': inference_time_full,
                'Inference Memory (MB)': inference_mem_full,
                'Inference Time Simulated (s)': None,
                'Inference Memory Simulated (MB)': None,
                'RMSE': rmse_saits_full,
                'MAE': mae_saits_full,
                'MAPE': mape_saits_full,
                'R2': r2_saits_full
            })

    # epoch losses for the final model
    plot_epoch_losses(loss_dict_saits, "SAITS")

    # computational metrics to a DataFrame and CSV
    comp_metrics_saits_df = pd.DataFrame(comp_metrics_saits)
    comp_metrics_saits_df.to_csv("saits_comp_metrics.csv", index=False)
    print("Saved computational metrics to 'saits_comp_metrics.csv'")

    return model_saits, comp_metrics_saits, loss_dict_saits