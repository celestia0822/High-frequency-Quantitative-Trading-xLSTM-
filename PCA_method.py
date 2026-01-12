import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

def apply_pca(data, variance_threshold=0.90):
    # Standardize the data
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    standardized_data = (data - data_mean) / data_std

    # Apply PCA
    pca = PCA(n_components=variance_threshold)
    pca_data = pca.fit_transform(standardized_data)

    # Get the number of components and explained variance ratio
    n_components = pca.n_components_
    explained_variance = pca.explained_variance_ratio_

    return pca_data, data_mean, data_std, explained_variance, n_components

def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = range(data.shape[1])
    vif_data["VIF"] = [variance_inflation_factor(data, i) for i in range(data.shape[1])]
    return vif_data
