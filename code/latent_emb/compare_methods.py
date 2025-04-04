# -------------------------
# TO compare all 3 of the advanced methods we can turn to here
#
#
# -------------------------


import pandas as pd
import matplotlib.pyplot as plt

# Load the metrics from all models
autoencoder_metrics = pd.read_csv("autoencoder_comp_metrics.csv")
saits_metrics = pd.read_csv("saits_comp_metrics.csv")
contrastive_metrics = pd.read_csv("contrastive_comp_metrics.csv")

# Add a 'Model' column to the Autoencoder and SAITS metrics (already present in contrastive metrics)
autoencoder_metrics['Model'] = 'Autoencoder'
saits_metrics['Model'] = 'SAITS'

# Combine the metrics
combined_metrics = pd.concat([autoencoder_metrics, saits_metrics, contrastive_metrics], ignore_index=True)

# Save the combined metrics
combined_metrics.to_csv("combined_comp_metrics_all.csv", index=False)
print("Saved combined computational metrics to 'combined_comp_metrics_all.csv'")

# Summarize test set metrics
test_metrics = combined_metrics[combined_metrics['Fold'] == 'Test'][['Model', 'RMSE', 'MAE', 'MAPE', 'R2']]
print("\n=== Test Set Performance Summary ===")
print(test_metrics)

# Plot a bar chart comparing RMSE
plt.figure(figsize=(8, 5))
plt.bar(test_metrics['Model'], test_metrics['RMSE'], color=['blue', 'orange', 'green'])
plt.xlabel("Model")
plt.ylabel("Test RMSE")
plt.title("Test RMSE Comparison: Autoencoder vs. SAITS vs. Contrastive")
plt.savefig("test_rmse_comparison_all.png")
plt.close()
print("Saved test RMSE comparison plot to 'test_rmse_comparison_all.png'")