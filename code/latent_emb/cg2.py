import itertools
import argparse
from cg1 import (train_contrastive_cv, load_data, MLP, ProjectionHead, ContrastiveTankDataset,
                 DataLoader, build_spatial_adjacency, build_temporal_edges, compute_degree_centrality,
                 graph_penalty, x_nce_loss, train_contrastive_model, impute_with_KNN, evaluate_imputation,
                 plot_epoch_losses, get_memory_usage)
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import time

def run_contrastive_final(best_params, df_full_normalized, df_train_normalized, df_test_normalized, tank_ids, fill_pcts, df, df_train_full, df_test_full, selected_features, tank_id_to_idx, device, comp_metrics, loss_dict, df_mean, df_std, use_graph_reg=True):
    lambda_graph, temperature, distance_threshold, lr = best_params
    if not use_graph_reg:
        lambda_graph = 0.0  # Disable graph regularization
        print("Running without graph regularization (lambda_graph=0.0)")

    print(f"\nTraining on full dataset with best hyperparameters: lambda_graph={lambda_graph}, temperature={temperature}, distance_threshold={distance_threshold}, lr={lr}")

    # graph structures
    spatial_adj_matrix = build_spatial_adjacency(df, tank_id_to_idx, distance_threshold=distance_threshold)
    temporal_adj_dict = build_temporal_edges(df)
    centrality_scores = compute_degree_centrality(spatial_adj_matrix)
    spatial_adj_matrix = spatial_adj_matrix.to(device)
    centrality_scores = centrality_scores.to(device)

    # training set
    train_dataset = ContrastiveTankDataset(df_train_normalized, tank_ids.iloc[:len(df_train_normalized)], mask_prob=0.2, jitter_std=0.01)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = MLP(input_dim=len(train_dataset.data[0]), hidden_dim=128).to(device)
    projection_head = ProjectionHead(input_dim=128, output_dim=64).to(device)

    # training time and memory measurement again
    start_time = time.time()
    start_mem = get_memory_usage()
    train_contrastive_model(model, projection_head, train_dataloader, tank_id_to_idx, spatial_adj_matrix, temporal_adj_dict, centrality_scores, num_epochs=30, lr=lr, lambda_graph=lambda_graph, device=device, temperature=temperature, loss_dict=loss_dict, model_name="Contrastive_Final")
    final_train_time = time.time() - start_time
    peak_mem = get_memory_usage()
    final_train_mem = peak_mem - start_mem

    # impute on test set
    start_time = time.time()
    start_mem = get_memory_usage()
    test_fill_pcts_imputed, test_embeddings = impute_with_KNN(df_test_normalized, tank_ids.iloc[len(df_train_normalized):len(df_train_normalized)+len(df_test_normalized)], df_test_full['fill_pct'], model, device)
    inference_time = time.time() - start_time
    peak_mem = get_memory_usage()
    inference_mem = peak_mem - start_mem

    # eval on the test set with simulated missingness
    non_missing_mask = df_test_full["fill_pct"].notna()
    if not non_missing_mask.any():
        print("[Contrastive] No non-missing fill_pct values in the test set to evaluate.")
    else:
        np.random.seed(42)
        mask_to_impute = non_missing_mask & (np.random.rand(len(df_test_full)) < 0.2)
        print(f"Simulating missingness: Masking {mask_to_impute.sum()} non-missing fill_pct values for evaluation.")

        df_test_simulated = df_test_full.copy()
        df_test_simulated.loc[mask_to_impute, "fill_pct"] = np.nan
        df_test_simulated_normalized = (df_test_simulated[selected_features] - df_mean) / df_std
        df_test_simulated_normalized = df_test_simulated_normalized.fillna(0)

        # inference time and memory for simulated missingness measurement
        start_time = time.time()
        start_mem = get_memory_usage()
        test_fill_pcts_imputed, _ = impute_with_KNN(df_test_simulated_normalized.values, df_test_simulated['tank_idx'], df_test_simulated['fill_pct'], model, device)
        inference_time_simulated = time.time() - start_time
        peak_mem = get_memory_usage()
        inference_mem_simulated = peak_mem - start_mem

        rmse, mae, mape, r2 = evaluate_imputation(df_test_full["fill_pct"], test_fill_pcts_imputed, mask_to_impute)
        if rmse is None:
            print("[Contrastive] Cannot evaluate: No artificially masked fill_pct values in test set.")
        else:
            print(f"[Contrastive] Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}, Test R2: {r2:.4f}")

            # Record computational metrics for the test set
            comp_metrics.append({
                'Hyperparams': f"lambda_graph={lambda_graph}, temp={temperature}, dist={distance_threshold}, lr={lr}",
                'Fold': 'Test',
                'Train Time (s)': final_train_time,
                'Train Memory (MB)': final_train_mem,
                'Inference Time (s)': inference_time,
                'Inference Memory (MB)': inference_mem,
                'Inference Time Simulated (s)': inference_time_simulated,
                'Inference Memory Simulated (MB)': inference_mem_simulated,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            })

    # Impute on the full dataset
    start_time = time.time()
    start_mem = get_memory_usage()
    fill_pcts_imputed, full_embeddings = impute_with_KNN(df_full_normalized, tank_ids, fill_pcts, model, device)
    inference_time_full = time.time() - start_time
    peak_mem = get_memory_usage()
    inference_mem_full = peak_mem - start_mem

    # eval on the full dataset with simulated missingness
    non_missing_mask_full = df["fill_pct"].notna()
    if not non_missing_mask_full.any():
        print("[Contrastive] No non-missing fill_pct values in the full dataset to evaluate.")
    else:
        np.random.seed(42)
        mask_to_impute_full = non_missing_mask_full & (np.random.rand(len(df)) < 0.2)
        print(f"Simulating missingness on full dataset: Masking {mask_to_impute_full.sum()} non-missing fill_pct values for evaluation.")

        df_full_simulated = df.copy()
        df_full_simulated.loc[mask_to_impute_full, "fill_pct"] = np.nan
        df_full_simulated_normalized = (df_full_simulated[selected_features] - df_mean) / df_std
        df_full_simulated_normalized = df_full_simulated_normalized.fillna(0)

        start_time = time.time()
        start_mem = get_memory_usage()
        fill_pcts_imputed, _ = impute_with_KNN(df_full_simulated_normalized.values, df_full_simulated['tank_idx'], df_full_simulated['fill_pct'], model, device)
        inference_time_simulated_full = time.time() - start_time
        peak_mem = get_memory_usage()
        inference_mem_simulated_full = peak_mem - start_mem

        rmse_full, mae_full, mape_full, r2_full = evaluate_imputation(df["fill_pct"], fill_pcts_imputed, mask_to_impute_full)
        if rmse_full is None:
            print("[Contrastive] Cannot evaluate: No artificially masked fill_pct values in full dataset.")
        else:
            print(f"[Contrastive] Full Dataset RMSE: {rmse_full:.4f}, Full Dataset MAE: {mae_full:.4f}, Full Dataset MAPE: {mape_full:.4f}, Full Dataset R2: {r2_full:.4f}")

            # Record computational metrics for the full dataset
            comp_metrics.append({
                'Hyperparams': f"lambda_graph={lambda_graph}, temp={temperature}, dist={distance_threshold}, lr={lr}",
                'Fold': 'Final',
                'Train Time (s)': final_train_time,
                'Train Memory (MB)': final_train_mem,
                'Inference Time (s)': inference_time_full,
                'Inference Memory (MB)': inference_mem_full,
                'Inference Time Simulated (s)': inference_time_simulated_full,
                'Inference Memory Simulated (MB)': inference_mem_simulated_full,
                'RMSE': rmse_full,
                'MAE': mae_full,
                'MAPE': mape_full,
                'R2': r2_full
            })

    # Plot epoch losses for the final run
    plot_epoch_losses(loss_dict, "Contrastive", save_path="epoch_losses_contrastive_final")

    return model, comp_metrics, loss_dict, full_embeddings, fill_pcts_imputed

'''
NOTE: To make latent space comparison more clear:

Try:

Increasing lambda_graph by an order of magnitude (e.g., 10× or 100×).

Reducing the contrastive temperature or adjusting the batch size so that the contrastive loss doesn’t overshadow the graph penalty.

If your distance_threshold is big, many tanks become “neighbors” in the adjacency matrix, so the penalty no longer forces truly distinct clusters to come together. 
Conversely, if it’s too small, very few edges exist, and the penalty has almost no effect.

Try:

Experiment with smaller thresholds (e.g., distance_threshold=2.0) or bigger thresholds, depending on how you expect your tanks to cluster geographically.

Check how many edges your adjacency matrix has. If it’s too dense or too sparse, you may need to tweak the threshold.

Try:

Changing the t-SNE parameters (e.g., lower perplexity, or set a different learning_rate, or run multiple seeds).

Using UMAP or PCA to see if differences become clearer.

Measuring “graph-awareness” in a more direct way (e.g., computing the average distance in embedding space between neighbors vs. non-neighbors).

graph_loss = graph_penalty(base_embeddings1, indices, tank_ids, ...)
But you might also consider using base_embeddings2 (the augmented view) or an average of both. 
If you only penalize one branch, the other might “escape” the adjacency constraints.

'''

if __name__ == "__main__":
    # Add command-line argument for toggling CV
    final_only = False

    csv_path = '/Users/finn/Desktop/UCL Masters/Data Science/Project/graphs/missingness/complete_panel_with_features_updated.csv'
    (df_full_normalized, df_train_normalized, df_test_normalized, tank_ids, fill_pcts, df_mean, df_std, df,
     df_train_full, df_test_full, selected_features, tank_id_to_idx) = load_data(csv_path)

    # Define hyperparameter grid for CV
    param_grid = {
        'lambda_graph': [1.0, 20.0, 50.0, 100.0],
        'temperature': [0.1, 10.0, 20.0],
        'distance_threshold': [5.0, 20.0],
        'lr': [1e-3, 5e-4, 1e-4]
    }
    param_combinations = list(itertools.product(
        param_grid['lambda_graph'],
        param_grid['temperature'],
        param_grid['distance_threshold'],
        param_grid['lr']
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_splits = 5

    # Initialize comp_metrics and loss_dict
    comp_metrics = []
    loss_dict = {}

    if final_only:
        # Use known best hyperparameters
        best_params = (1.0, 0.1, 5.0, 1e-3)  # lambda_graph, temperature, distance_threshold, lr KNOWN BEST
        print("\n=== Skipping CV and using known best hyperparameters ===")
        print(f"Best hyperparameters: lambda_graph={best_params[0]}, temperature={best_params[1]}, distance_threshold={best_params[2]}, lr={best_params[3]}")
    else:
        # Run CV to find the best hyperparameters
        print("\n=== Running Contrastive Model CV ===")
        best_params, comp_metrics, loss_dict = train_contrastive_cv(
            param_combinations, n_splits, df_train_normalized, df_train_full, tank_ids, fill_pcts,
            selected_features, tank_id_to_idx, device, df_mean, df_std
        )

    # Run the final model with graph regularization
    print("\n=== Running Contrastive Model Final (With Graph Regularization) ===")
    model_with_reg, comp_metrics, loss_dict, embeddings_with_reg, fill_pcts_imputed_with_reg = run_contrastive_final(
        best_params, df_full_normalized, df_train_normalized, df_test_normalized, tank_ids, fill_pcts,
        df, df_train_full, df_test_full, selected_features, tank_id_to_idx, device, comp_metrics, loss_dict,
        df_mean, df_std, use_graph_reg=True
    )

    # Run the final model without graph regularization
    print("\n=== Running Contrastive Model Final (Without Graph Regularization) ===")
    model_without_reg, comp_metrics, loss_dict, embeddings_without_reg, fill_pcts_imputed_without_reg = run_contrastive_final(
        best_params, df_full_normalized, df_train_normalized, df_test_normalized, tank_ids, fill_pcts,
        df, df_train_full, df_test_full, selected_features, tank_id_to_idx, device, comp_metrics, loss_dict,
        df_mean, df_std, use_graph_reg=False
    )

    # Generate scatter plots of the latent space embeddings
    print("\n=== Generating Latent Space Scatter Plots ===")
    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_with_reg_2d = tsne.fit_transform(embeddings_with_reg)
    embeddings_without_reg_2d = tsne.fit_transform(embeddings_without_reg)

    # Create scatter plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot with graph regularization
    scatter1 = ax1.scatter(embeddings_with_reg_2d[:, 0], embeddings_with_reg_2d[:, 1], c=tank_ids, cmap='tab20', alpha=0.6, s=50)
    ax1.set_title("With Graph Regularization")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")
    fig.colorbar(scatter1, ax=ax1, label='Tank ID')

    # Plot without graph regularization
    scatter2 = ax2.scatter(embeddings_without_reg_2d[:, 0], embeddings_without_reg_2d[:, 1], c=tank_ids, cmap='tab20', alpha=0.6, s=50)
    ax2.set_title("Without Graph Regularization")
    ax2.set_xlabel("t-SNE Component 1")
    ax2.set_ylabel("t-SNE Component 2")
    fig.colorbar(scatter2, ax=ax2, label='Tank ID')

    plt.tight_layout()
    plt.savefig("latent_space_comparison_contrastive.png")
    plt.close()
    print("Saved latent space comparison plot to 'latent_space_comparison_contrastive.png'")

    # Save computational metrics
    comp_metrics_df = pd.DataFrame(comp_metrics)
    comp_metrics_df['Model'] = 'Contrastive'
    comp_metrics_df.to_csv("contrastive_comp_metrics.csv", index=False)
    print("Saved computational metrics to 'contrastive_comp_metrics.csv'")

    # Save the imputed dataset (using the model with graph regularization)
    df['fill_pct_imputed'] = fill_pcts_imputed_with_reg
    df.to_csv("imputed_dataset_contrastive.csv", index=False)
    print("Saved imputed dataset using Contrastive method.")