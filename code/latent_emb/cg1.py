# -------------------------
# Time for the model tailored for this dataset -> i.e. incorporating graph regularization with a latent embedding model
# The goal is to use the domain knowledge of the proximity of the tanks to pull their latent representations closer
# together for enhanced imputation.
#
# -------------------------


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import KDTree
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import TimeSeriesSplit
import itertools
import time
import psutil
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from imputation.imp_prep import plot_epoch_losses, track_losses

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Memory in MB

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    selected_features = [
        'distance_to_nearest_port_lag3', 'wind_speed_roll_mean', 'fill_pct', 'fill_pct_lag1',
        'fill_pct_lag2', 'fill_pct_lag3', 'fill_pct_roll_mean', 'fill_pct_roll_std',
        'distance_to_nearest_tank_id_lag3', 'temperature_avg', 'temperature_min_roll_mean',
        'wti_brent_spread_roll_mean', 'wind_dir_roll_mean', 'wti_price', 'season_numeric'
    ]

    # Parse Location to extract lat and lon
    if 'Location' in df.columns:
        def parse_location(loc):
            if pd.isna(loc) or 'POINT' not in loc:
                return np.nan, np.nan
            coords = loc.replace('POINT(', '').replace(')', '').split()
            lon, lat = float(coords[0]), float(coords[1])
            return lat, lon
        df[['lat', 'lon']] = df['Location'].apply(parse_location).apply(pd.Series)

    # Ensure season_numeric is created if season exists
    if 'season' in df.columns and 'season_numeric' not in df.columns:
        df['season_numeric'] = df['season'].map({'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}).astype(float)

    # Sort by tank_id and imaging_time
    if 'tank_id' in df.columns and 'imaging_time' in df.columns:
        df['imaging_time'] = pd.to_datetime(df['imaging_time'], errors='coerce')
        df = df.sort_values(['tank_id', 'imaging_time']).reset_index(drop=True)

    # Map tank_ids to integer indices
    tank_id_to_idx = {tid: idx for idx, tid in enumerate(df['tank_id'].unique())}
    df['tank_idx'] = df['tank_id'].map(tank_id_to_idx)

    # Feature engineering for lag and rolling features
    for feature in selected_features:
        if feature not in df.columns:
            if 'lag' in feature:
                base_col = feature.split('_lag')[0]
                lag_num = int(feature.split('_lag')[1])
                if base_col in df.columns:
                    df[feature] = df.groupby('tank_id')[base_col].shift(lag_num)
            elif 'roll_mean' in feature:
                base_col = feature.split('_roll_mean')[0]
                if base_col in df.columns:
                    df[feature] = (df.groupby('tank_id')[base_col]
                                   .rolling(window=7, min_periods=1)
                                   .mean()
                                   .reset_index(level=0, drop=True))
            elif 'roll_std' in feature:
                base_col = feature.split('_roll_std')[0]
                if base_col in df.columns:
                    df[feature] = (df.groupby('tank_id')[base_col]
                                   .rolling(window=7, min_periods=1)
                                   .std()
                                   .reset_index(level=0, drop=True))

    # Final check for missing features
    for feature in selected_features:
        if feature not in df.columns:
            print(f"Somehow we're missing {feature}! Likely a lag or rolling feature couldn't be created.")
    print('"A" ok')

    # Select only the specified features
    df_selected = df[selected_features].copy()

    # Train-test split (70/30) preserving temporal order
    n_rows = len(df_selected)
    train_size = int(n_rows * 0.7)
    train_idx = np.arange(0, train_size)
    test_idx = np.arange(train_size, n_rows)
    df_train = df_selected.iloc[train_idx].copy()
    df_test = df_selected.iloc[test_idx].copy()
    df_train_full = df.iloc[train_idx].copy()
    df_test_full = df.iloc[test_idx].copy()

    # Normalize the data (fit on training data only)
    df_mean = df_train.mean()
    df_std = df_train.std()
    df_std = df_std.replace(0, 1e-6)  # Avoid division by zero
    df_train_normalized = (df_train - df_mean) / df_std
    df_test_normalized = (df_test - df_mean) / df_std
    df_train_normalized = df_train_normalized.fillna(0)
    df_test_normalized = df_test_normalized.fillna(0)

    # Normalize the entire dataset for final imputation
    df_full_normalized = (df_selected - df_mean) / df_std
    df_full_normalized = df_full_normalized.fillna(0)

    tank_ids = df['tank_idx']
    fill_pcts = df['fill_pct']
    return (df_full_normalized.values, df_train_normalized.values, df_test_normalized.values,
            tank_ids, fill_pcts, df_mean, df_std, df, df_train_full, df_test_full, selected_features, tank_id_to_idx)

class ContrastiveTankDataset(Dataset):
    def __init__(self, data, tank_ids, mask_prob=0.2, jitter_std=0.01):
        self.data = data
        self.tank_ids = tank_ids
        self.mask_prob = mask_prob
        self.jitter_std = jitter_std
        self.num_features = data.shape[1]
        self.indices = torch.arange(len(data), dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        tank_id = self.tank_ids.iloc[idx] if isinstance(self.tank_ids, pd.Series) else self.tank_ids[idx]
        x1 = self.augment(x)
        x2 = self.augment(x)
        return (torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32),
                torch.tensor(tank_id, dtype=torch.long), self.indices[idx])
    
    def augment(self, x):
        x_aug = x.copy()
        mask_count = int(self.mask_prob * self.num_features)
        mask_indices = np.random.choice(self.num_features, mask_count, replace=False)
        for mi in mask_indices:
            x_aug[mi] = 0.0
        noise = np.random.normal(0, self.jitter_std, size=self.num_features).astype(np.float32)
        x_aug += noise
        return x_aug
    
class MLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
    def forward(self, x):
        return self.net(x)
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=64):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, output_dim))
    def forward(self, x):
        return self.head(x)

def build_spatial_adjacency(df, tank_id_to_idx, distance_threshold=10.0):
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    tank_ids = df['tank_id'].unique()
    tank_locs = {}
    coords = []
    for tid in tank_ids:
        row = df[df['tank_id'] == tid].iloc[0]
        lat, lon = row['lat'], row['lon']
        if pd.isna(lat) or pd.isna(lon):
            continue
        tank_locs[tid] = (lat, lon)
        coords.append([lat, lon])
    
    tank_ids = list(tank_locs.keys())
    coords = np.array(coords)
    
    tree = KDTree(coords)
    euclidean_threshold = distance_threshold / 111.0
    pairs = tree.query_ball_tree(tree, r=euclidean_threshold)
    
    num_tanks = len(tank_id_to_idx)
    adj_matrix = torch.zeros((num_tanks, num_tanks))
    
    for i, tidA in enumerate(tank_ids):
        latA, lonA = tank_locs[tidA]
        candidates = pairs[i]
        for j in candidates:
            if i == j:
                continue
            tidB = tank_ids[j]
            latB, lonB = tank_locs[tidB]
            dist = haversine_distance(latA, lonA, latB, lonB)
            if dist < distance_threshold:
                idxA = tank_id_to_idx[tidA]
                idxB = tank_id_to_idx[tidB]
                adj_matrix[idxA, idxB] = 1
                adj_matrix[idxB, idxA] = 1
    
    return adj_matrix

def build_temporal_edges(df):
    temporal_adj_dict = {}
    df_sorted = df.sort_values(['tank_id', 'imaging_time']).reset_index()
    for tank_id, group in df_sorted.groupby('tank_id'):
        indices = group.index.tolist()
        for i in range(len(indices) - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            if idx1 not in temporal_adj_dict:
                temporal_adj_dict[idx1] = []
            if idx2 not in temporal_adj_dict:
                temporal_adj_dict[idx2] = []
            temporal_adj_dict[idx1].append(idx2)
            temporal_adj_dict[idx2].append(idx1)  # Undirected
    return temporal_adj_dict

def compute_degree_centrality(adj_matrix):
    centrality = adj_matrix.sum(dim=1)
    max_degree = centrality.max()
    if max_degree > 0:
        centrality = centrality / max_degree
    return centrality

def graph_penalty(z, batch_indices, tank_ids, tank_id_to_idx, spatial_adj_matrix, temporal_adj_dict, centrality_scores, lambda_graph=1.0):
    z = F.normalize(z, dim=1)
    batch_size = z.shape[0]
    penalty = torch.tensor(0.0, device=z.device)

    # Spatial penalty: fully vectorized
    tank_indices = tank_ids
    spatial_mask = spatial_adj_matrix[tank_indices][:, tank_indices]
    centrality_batch = centrality_scores[tank_indices]
    centrality_weights = (centrality_batch.unsqueeze(1) + centrality_batch.unsqueeze(0)) / 2 + 1e-6

    z_expanded_i = z.unsqueeze(1)
    z_expanded_j = z.unsqueeze(0)
    squared_dists = (z_expanded_i - z_expanded_j).pow(2).sum(dim=2)
    spatial_penalty = (spatial_mask * centrality_weights * squared_dists).sum()
    spatial_count = spatial_mask.sum().item()

    # Temporal penalty: vectorized with precomputed index
    temporal_pairs = []
    for i, idx in enumerate(batch_indices):
        idx = idx.item()
        temporal_neighbors = temporal_adj_dict.get(idx, [])
        for neighbor_idx in temporal_neighbors:
            if neighbor_idx in batch_indices:
                j = (batch_indices == neighbor_idx).nonzero(as_tuple=True)[0].item()
                temporal_pairs.append((i, j))
    
    temporal_penalty = torch.tensor(0.0, device=z.device)
    temporal_count = 0
    if temporal_pairs:
        temporal_pairs = torch.tensor(temporal_pairs, device=z.device)
        idx1, idx2 = temporal_pairs[:, 0], temporal_pairs[:, 1]
        temporal_dists = (z[idx1] - z[idx2]).pow(2).sum(dim=1)
        temporal_penalty = temporal_dists.sum()
        temporal_count = len(temporal_pairs)

    total_count = spatial_count + temporal_count
    if total_count > 0:
        penalty = (spatial_penalty + temporal_penalty) / (total_count + 1e-6)
    return lambda_graph * penalty

def x_nce_loss(embeddings1, embeddings2, temperature=10.0):
    batch_size = embeddings1.size(0)
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)
    sim_matrix = torch.mm(embeddings1, embeddings2.t()) / temperature
    labels = torch.arange(batch_size).to(embeddings1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def train_contrastive_model(model, projection_head, dataloader, tank_id_to_idx, spatial_adj_matrix, temporal_adj_dict, centrality_scores, num_epochs=30, lr=1e-3, lambda_graph=1.0, device='cpu', temperature=10.0, loss_dict=None, model_name="Contrastive"):
    model = model.to(device)
    projection_head = projection_head.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projection_head.parameters()), lr=lr)
    model.train()
    projection_head.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_contrastive_loss = 0
        total_graph_loss = 0
        for x1, x2, tank_ids, indices in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            tank_ids = tank_ids.to(device)
            indices = indices.to(device)
            optimizer.zero_grad()
            base_embeddings1 = model(x1)
            base_embeddings2 = model(x2)
            proj_embeddings1 = projection_head(base_embeddings1)
            proj_embeddings2 = projection_head(base_embeddings2)
            contrastive_loss = x_nce_loss(proj_embeddings1, proj_embeddings2, temperature=temperature)
            graph_loss = graph_penalty(base_embeddings1, indices, tank_ids, tank_id_to_idx, spatial_adj_matrix, temporal_adj_dict, centrality_scores, lambda_graph)
            loss = contrastive_loss + graph_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_graph_loss += graph_loss.item()
        avg_loss = total_loss / len(dataloader)
        if loss_dict is not None:
            track_losses(model_name, "Current", epoch, avg_loss, loss_dict)
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {avg_loss:.4f}, "
              f"Contrastive Loss: {total_contrastive_loss/len(dataloader):.4f}, "
              f"Graph Loss: {total_graph_loss/len(dataloader):.4f}")

def impute_with_KNN(data, tank_ids, fill_pcts, model, device='cpu'):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(len(data)):
            x = torch.tensor(data[i], dtype=torch.float32).to(device)
            emb = model(x.unsqueeze(0)).cpu().numpy()
            embeddings.append(emb[0])
    embeddings = np.array(embeddings)
    known_mask = ~fill_pcts.isna()
    unknown_mask = fill_pcts.isna()

    X_train = embeddings[known_mask]
    y_train = fill_pcts[known_mask].values
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    X_test = embeddings[unknown_mask]
    imputed_values = knn.predict(X_test)

    fill_pcts_imputed = fill_pcts.copy()
    fill_pcts_imputed[unknown_mask] = imputed_values
    return fill_pcts_imputed, embeddings

def evaluate_imputation(original_series, imputed_series, mask):
    if not mask.any():
        return None, None, None, None
    rmse = np.sqrt(mean_squared_error(original_series[mask], imputed_series[mask]))
    mae = mean_absolute_error(original_series[mask], imputed_series[mask])
    mask_nonzero = mask & (original_series != 0)
    if mask_nonzero.sum() > 0:
        mape = np.mean(np.abs((original_series[mask_nonzero] - imputed_series[mask_nonzero]) / original_series[mask_nonzero])) * 100
    else:
        mape = None
    r2 = r2_score(original_series[mask], imputed_series[mask])
    return rmse, mae, mape, r2

def plot_latent_space(embeddings, tank_ids, title, filename):
    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=tank_ids, cmap='tab20', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Tank ID')
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(filename)
    plt.close()
    print(f"Saved latent space plot to '{filename}'")

def train_contrastive_cv(param_combinations, n_splits, df_train_normalized, df_train_full, tank_ids, fill_pcts, selected_features, tank_id_to_idx, device, df_mean, df_std):
    comp_metrics = []
    loss_dict = {}
    best_params = None
    best_avg_rmse = float('inf')

    # Walk-forward cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    df_sorted = df_train_full.sort_values('imaging_time').reset_index()
    data_sorted = df_train_normalized[df_sorted.index]
    tank_ids_sorted = tank_ids.iloc[df_sorted.index]
    fill_pcts_sorted = fill_pcts.iloc[df_sorted.index]

    for params in param_combinations:
        lambda_graph, temperature, distance_threshold, lr = params
        print(f"\nTesting hyperparameters: lambda_graph={lambda_graph}, temperature={temperature}, distance_threshold={distance_threshold}, lr={lr}")

        print("Building spatial adjacencies...")
        spatial_adj_matrix = build_spatial_adjacency(df_train_full, tank_id_to_idx, distance_threshold=distance_threshold)
        print("Building temporal edges...")
        temporal_adj_dict = build_temporal_edges(df_train_full)
        print("Computing centrality metrics...")
        centrality_scores = compute_degree_centrality(spatial_adj_matrix)
        spatial_adj_matrix = spatial_adj_matrix.to(device)
        centrality_scores = centrality_scores.to(device)

        rmse_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data_sorted)):
            print(f"\nFold {fold+1}/{n_splits}")
            
            train_data = data_sorted[train_idx]
            test_data = data_sorted[test_idx]
            train_tank_ids = tank_ids_sorted.iloc[train_idx]
            test_tank_ids = tank_ids_sorted.iloc[test_idx]
            train_fill_pcts = fill_pcts_sorted.iloc[train_idx]
            test_fill_pcts = fill_pcts_sorted.iloc[test_idx]
            test_df = df_sorted.iloc[test_idx].copy()

            train_dataset = ContrastiveTankDataset(train_data, train_tank_ids, mask_prob=0.2, jitter_std=0.01)
            train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            model = MLP(input_dim=len(train_dataset.data[0]), hidden_dim=128).to(device)
            projection_head = ProjectionHead(input_dim=128, output_dim=64).to(device)

            # Measure training time and memory
            start_time = time.time()
            start_mem = get_memory_usage()
            print("Training contrastive model...")
            train_contrastive_model(model, projection_head, train_dataloader, tank_id_to_idx, spatial_adj_matrix, temporal_adj_dict, centrality_scores, num_epochs=20, lr=lr, lambda_graph=lambda_graph, device=device, temperature=temperature, loss_dict=loss_dict, model_name=f"Contrastive_Fold_{fold+1}")
            train_time = time.time() - start_time
            peak_mem = get_memory_usage()
            train_mem = peak_mem - start_mem

            # Measure inference time and memory for the first imputation (on test set)
            start_time = time.time()
            start_mem = get_memory_usage()
            test_fill_pcts_imputed, _ = impute_with_KNN(test_data, test_tank_ids, test_fill_pcts, model, device)
            inference_time = time.time() - start_time
            peak_mem = get_memory_usage()
            inference_mem = peak_mem - start_mem

            non_missing_mask = test_df["fill_pct"].notna()
            if not non_missing_mask.any():
                print(f"Skipping Fold {fold+1}: No non-missing fill_pct values in the test set to evaluate.")
                continue

            np.random.seed(42 + fold)
            mask_to_impute = non_missing_mask & (np.random.rand(len(test_df)) < 0.2)
            print(f"Simulating missingness: Masking {mask_to_impute.sum()} non-missing fill_pct values for evaluation.")

            test_df_simulated = test_df.copy()
            test_df_simulated.loc[mask_to_impute, "fill_pct"] = np.nan
            test_df_simulated_normalized = (test_df_simulated[selected_features] - df_mean) / df_std
            test_df_simulated_normalized = test_df_simulated_normalized.fillna(0)

            # Measure inference time and memory for the second imputation (simulated missingness)
            start_time = time.time()
            start_mem = get_memory_usage()
            test_fill_pcts_imputed, _ = impute_with_KNN(test_df_simulated_normalized.values, test_df_simulated['tank_idx'], test_df_simulated['fill_pct'], model, device)
            inference_time_simulated = time.time() - start_time
            peak_mem = get_memory_usage()
            inference_mem_simulated = peak_mem - start_mem

            rmse, mae, mape, r2 = evaluate_imputation(test_df["fill_pct"], test_fill_pcts_imputed, mask_to_impute)
            if rmse is None:
                print("Cannot evaluate: No artificially masked fill_pct values in test set.")
            else:
                print(f"Fold {fold+1} - Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}, Test R2: {r2:.4f}")
                rmse_scores.append(rmse)

                # Record computational metrics for this fold
                comp_metrics.append({
                    'Hyperparams': f"lambda_graph={lambda_graph}, temp={temperature}, dist={distance_threshold}, lr={lr}",
                    'Fold': fold + 1,
                    'Train Time (s)': train_time,
                    'Train Memory (MB)': train_mem,
                    'Inference Time (s)': inference_time,
                    'Inference Memory (MB)': inference_mem,
                    'Inference Time Simulated (s)': inference_time_simulated,
                    'Inference Memory Simulated (MB)': inference_mem_simulated,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'R2': r2
                })

        if rmse_scores:
            avg_rmse = np.mean(rmse_scores)
            print(f"Average RMSE for params {params}: {avg_rmse:.4f} ± {np.std(rmse_scores):.4f}")
            if avg_rmse < best_avg_rmse:
                best_avg_rmse = avg_rmse
                best_params = params

    print("\nBest hyperparameters:")
    print(f"lambda_graph={best_params[0]}, temperature={best_params[1]}, distance_threshold={best_params[2]}, lr={best_params[3]}")
    print(f"Best average RMSE: {best_avg_rmse:.4f}")

    # Plot epoch losses for CV
    plot_epoch_losses(loss_dict, "Contrastive", save_path="epoch_losses_contrastive_cv")

    return best_params, comp_metrics, loss_dict