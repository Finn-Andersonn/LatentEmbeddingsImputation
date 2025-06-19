# Enhancing Missing Data Imputation Through Network-Driven Latent Embeddings

## TL;DR
- **What this is**: Missing data is a difficult problem, especially when it's severe, as in the Louisiana Oil Dataset. To better reflect spatial proximity and temporal importance, we transform features into hidden patterns (latent embeddings) that capture structure not directly visible in the raw data.
- **Why this matters**: Existing complex methods (like SAITS and VAE) can memorize patterns without comprehending true underlying relationships (overfit) in these conditions without strong adjustments  (regularization), motivating the development of our third alternative.
- **Why this is cool**: With our third model, we outperform existing advanced methods, offering a more robust solution to the missing data problem.
## Abstract
Multivariate time series imputation remains a pivotal problem in advanced data analysis. Time series data is ubiquitous, stretching across healthcare, finance, meteorology, and, in this paper, oil tank data. However, recordings are not always complete: instrument malfunctions, data corruption, communication failures, weather restrictions, or even regulatory changes produce missing values. One widespread solution to this issue is imputation: the process of replacing missing points in a dataset based on observed values. In this paper, I implement a Latent Embedding Contrastive Learning model to solve this problem, using a Louisiana Tank Farm Dataset as a case study.

## Findings
First, we demonstrated that graph regularization provides a slight improvement over contrastive learning alone when features with ≈ 0.1 Spearman correlation or mutual information are used, as seen in the slight reduction in RMSE on the Louisiana Tank Farm dataset. Second, our CL×GR outperformed other state- of-the-art models, including the SAITS and autoencoder, across RMSE, MAE, MAPE, and R2. Thirdly, we achieved a ∼10% reduction in absolute error on an extremely sparse, weak-signal panel, and produced a more faithful latent geometry for downstream analysis of the Louisiana Tank Farm dataset. This provides evidence that graph-aware contrastive learning is a worthwhile addition to the imputation toolkit.

