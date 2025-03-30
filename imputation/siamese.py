# ----------------------------------------------------------------------------
# In the industrial tank network, many operational values (fill percentage or volumes) are missing, but we have reliable spatial and temporal information
# This loss is largely due to satellite imaging infrequency (refer to corr.py or causality.py). 
# We can use a Siamese Neural Network to impute the missing values (refer to imputation.py to see the other methods and their losses).
#   -> A Siamese network can learn a latent space in which tanks with similar operational behaviors (even when some measurements are missing) 
#      are embedded close together.
#   -> Once this space is learned, we can impute missing data using a nearest-neighbor approach in the latent space
#   -> useful when standard imputation methods (like mean or KNN on raw features) might fail due to large gaps or non-linear relationships (again imputation.py)
#   -> NOT a regression. Regressions typically predict a single variable (e.g. fill level) from a set of input features.
#      By contrast, latent space models (VAE/Siamese) aim to learn a hidden representation (embedding) that captures underlying structure in the data (car e.g.!!)
#   -> NOT VAE but Siamese bc the goal isnt generation but direct similarity learning
#   -> We use latent models for representation learning, which allows us to conduct imputation
#
#
# Standard methods assume linear relationships or rely on fixed statistics of the data. They might not capture the complex interactions between spatial, 
# temporal, and operational features.
#   -> We'll prove this
#
# Variational Autoencoders
#   -> For argument sake say Autoencoders did well
#   -> VAEs learn a probabilistic latent space, enforcing continuity and completeness by regularizing the latent distribution toward a Gaussian via KL divergence.
#   -> VAEs are not primarily designed to optimize pairwise similarity. In contrast, Siamese networks directly optimize for similarity, which is crucial for 
#      accurately imputing missing values based on “neighbors” in latent space.
#   -> That said, the latent space properties of VAEs—continuity and completeness—are desirable. One could consider a hybrid approach that incorporates aspects of VAEs 
#      (ensuring a well-regularized latent space) with contrastive or triplet losses to directly target similarity.
#
#
# Integrating Graphs
#   -> While a Siamese network produces a latent embedding based on feature similarity, we also have reliable spatial data.
#   -> Tanks that are geographically close might share similar operational conditions due to environmental or managerial factors.
#   ->  Constructing a graph where nodes represent tanks and edges represent either spatial proximity or a combined similarity (e.g., using both latent distances and 
#       physical distances) allows us to leverage known relational structure in the network.
#   -> We'll test Euclidean distance, euclidean centrality, etc. <- a bunch of similarity measures
#
#
# Siamese Network
#   -> We need a similarity measure in this to inform the Siamese Network
#      ~ Euclidean distance, cosine similarity, or even learned Mahalanobis distances.
#      ~ compare these by evaluating the quality of imputation (using metrics like MAE or RMSE) or by examining clustering quality in the latent space.
#      ~ scatterplots or performance curves
# 
# Problem Setup:
#      -> Problem setup: dataset has missing values (e.g., fill_pct) due to temporal skew (pre-noon, non-summer bias). 
#         Traditional imputation (e.g., mean, k-NN) ignores complex relationships like spatial proximity or operational similarity.
#         Similarity-based methods (Siamese, contrastive) learn a latent space where "similar" tanks are close, enabling better 
#         imputation via k-NN in that space. This is especially powerful for industrial data with spatial and temporal patterns.
#         Contrastive learning (e.g., SimCLR) scales better than Siamese networks by leveraging large batches of data and augmentations, 
#         avoiding the need for explicit pairwise labeling. It generalizes better across diverse features (e.g., max_vol, imaging_time, region), 
#         which is critical for the dataset’s heterogeneity.
#      -> You can validate this jump by comparing Siamese vs. contrastive embeddings on a proxy task (e.g., clustering tanks by region). 
#         If contrastive learning improves silhouette score by a statistically significant margin (p < 0.05 via t-test), it justifies the upgrade.
#
# Roadmap:
#   1. Feature Expansion
#      -> Add weather (temp, cloud cover), EIA reporting dates, season, oil price, distance to port.
#      -> Cap at 5 new features; monitor condition number (<10^4).
#   2. Correlation and Causation
#      -> Pearson/Spearman for imaging_time vs. weather/EIA dates; report p-values, plot 5%/1% significance bounds.
#      -> Granger causality (exploratory) for EIA → imaging_time; include EIA dates if significant (p < 0.05).
#      -> Check covariance: Frobenius norm, condition number vs. q/p, regression error vs. q/p.
#      -> Compute Pearson/Spearman correlations to justify features (e.g., fill_pct vs. imaging_time); report p-values.
#      -> Refer to corr.py & causality.py
#   3. Baseline Imputation Weaknesses
#      -> Test mean/median, k-NN, MICE on holdout set (MAE, RMSE); compare with t-tests (p < 0.05).
#      -> Simple methods like PCA or autoencoders don’t explicitly optimize for similarity, which is central to your imputation (k-NN in latent space) 
#         and RAG (retrieval) goals.
#      -> Graph-based methods alone (e.g., GNNs without contrastive learning) lack the flexibility to learn a general-purpose similarity metric across all features.
#   4. Quick-Fix Missing Values
#      -> Median imputation per region for operational dataset; note limitations (e.g., temporal skew).
#      -> Research tests if advanced methods beat baselines (target: 10% MAE/RMSE reduction).
#       -> a quick fix for missing data first just to get an operational dataset. 
#          However, this doesn’t always capture the true complexity of the data—especially if large portions are missing
#   5. Train Contrastive Learning (Primary) and Siamese Network (Baseline)
#      -> Contrastive (SimCLR-inspired): MLP encoder f(·), 2-layer projection head g(·); augmentations (jitter fill_pct, perturb imaging_time, mask region).
#      -> NT-Xent loss on augmented pairs (e.g., same farm, different times); Siamese: contrastive loss on pairwise inputs for comparison.
#      -> Add graph regularization: L[total] = L[NT-Xent] + lambda1 * L[graph]; L[graph] pulls spatially close tanks together.
#      -> Add KL-divergence: L[total] += lambda2 * L[KL] for smoothness; visualize embeddings (only NT-Xent, only graph, only KL, combined).
#      -> ensuring your embeddings are well-behaved for downstream tasks like imputation and retrieval.
#      -> Validate: Silhouette score, t-test for contrastive vs. Siamese (p < 0.05).
#   6. Graph Construction
#      -> Nodes: tanks; edges: d[combined] = alpha * d[latent] + (1-alpha) * d[spatial]; tune alpha.
#      -> Validate: Edge density, clustering coefficient, correlation with farm_type.
#   7. Imputation with Embeddings
#      -> k-NN in latent space (contrastive/Siamese); impute missing fill_pct as neighbor average.
#      -> Evaluate: MAE/RMSE on holdout; t-test vs. baselines (p < 0.05).
#   8. RAG Extension
#      -> Use embeddings for similarity-based retrieval of search history/website context to optimize AI prompts.
#      -> Validate: Precision/recall on synthetic search dataset.
#   9. Visualizations
#      -> Embedding visualization: After training, extract the latent embeddings for each tank
#      -> Reduce dimensionality to 2D using PCA
#      -> Plot scatters, colouring points by a meaningful feature (e.g. region, farm_type, or binned imaging like "morning" or "afternoon")
#           ~ Plot for L[contrastive] only: Embeddings should cluster based on operation similarity (e.g. tanks with similiar fill_pct or max_vol)
#           ~ Plot for L[graph] only: Embeddings will cluster around spatial proximity (e.g. same farm/region) potentially losing operational distinctions
#           ~ Plot for L[KL] only: Embeddings will collapse towards a Gaussian Blob (like VAE over MNIST for KL only plot) losing most structure
#           ~ Combined: A balance --- clusters reflecting both operational similarity and spatial relationships, with smoother transitions due to KL
#           ~ lambda[1] and lambda[2] tuned through grid search, optimizing for downstream imputation performance (RMSE/MAE)
# ----------------------------------------------------------------------------