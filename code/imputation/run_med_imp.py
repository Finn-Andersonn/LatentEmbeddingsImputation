# ----------------------------------------
# Running autoencoder and SAITS models
# ----------------------------------------

from imp_prep import walk_forward_split, prep
from autoencoder import train_autoencoder_cv, run_autoencoder_final
from SAITS import train_saits_cv, run_saits_final
import pandas as pd

# setup
final_features, df_full_normalized, df_train_normalized, df_test_normalized, df_train, df_test, df_std, df_mean, df = prep()
input_dim = len(final_features)
fill_pct_idx = final_features.index('fill_pct')
n_splits = 5
splits = walk_forward_split(df_train_normalized, n_splits=n_splits)

# Option to rerun CV or use best hyperparameters directly
rerun_cv = False  # Set to True to rerun CV, False to use best hyperparameters directly

# Autoencoder hyperparameters
encoding_dims = [4, 8, 12]
best_encoding_dim = 8  # From previous grid search

# SAITS hyperparameters
d_model_options = [16, 32, 64]
nhead_options = [4, 4, 8]
num_layers_options = [2, 4, 6]
best_saits_params = (32, 4, 2)  # From previous grid search

# Autoencoder
print("\n=== Running Autoencoder ===")
if rerun_cv:
    # CV to find the best encoding_dim
    best_encoding_dim, comp_metrics_ae, loss_dict_ae = train_autoencoder_cv(
        encoding_dims, splits, n_splits, df_train_normalized, df_train, input_dim, df_std, df_mean, fill_pct_idx
    )
else:
    # Using the best encoding_dim directly
    comp_metrics_ae = []
    loss_dict_ae = {}
    print(f"Using best encoding_dim from previous grid search: {best_encoding_dim}")

# final Autoencoder model
model_ae, comp_metrics_ae, loss_dict_ae = run_autoencoder_final(
    best_encoding_dim, df_train_normalized, df_train, df_test_normalized, df_test, df_full_normalized, df,
    input_dim, df_std, df_mean, fill_pct_idx, comp_metrics_ae, loss_dict_ae
)

# SAITS
print("\n=== Running SAITS ===")
if rerun_cv:
    # CV to find the best SAITS parameters
    best_saits_params, comp_metrics_saits, loss_dict_saits = train_saits_cv(
        d_model_options, nhead_options, num_layers_options, n_splits, splits,
        df_train_normalized, df_train, input_dim, df_std, df_mean, fill_pct_idx
    )
else:
    # using best SAITS parameters directly
    comp_metrics_saits = []
    loss_dict_saits = {}
    print(f"Using best SAITS params from previous grid search: d_model={best_saits_params[0]}, nhead={best_saits_params[1]}, num_layers={best_saits_params[2]}")

# the final SAITS model
model_saits, comp_metrics_saits, loss_dict_saits = run_saits_final(
    best_saits_params, df_train_normalized, df_train, df_test_normalized, df_test, df_full_normalized, df,
    input_dim, df_std, df_mean, fill_pct_idx, comp_metrics_saits, loss_dict_saits
)

# Combine computational metrics from both models
comp_metrics_ae_df = pd.DataFrame(comp_metrics_ae)
comp_metrics_saits_df = pd.DataFrame(comp_metrics_saits)
comp_metrics_ae_df['Model'] = 'Autoencoder'
comp_metrics_saits_df['Model'] = 'SAITS'
combined_metrics = pd.concat([comp_metrics_ae_df, comp_metrics_saits_df], ignore_index=True)
combined_metrics.to_csv("combined_comp_metrics.csv", index=False)
print("Saved combined computational metrics to 'combined_comp_metrics.csv'")