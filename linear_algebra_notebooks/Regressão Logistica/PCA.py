import numpy as np
def compute_pca(data, n_components):
    data_meaned = data - np.mean(data, axis=0)
    std = np.std(data_meaned, axis=0)
    data_meaned /= std
    covariance_matrix = data_meaned.T.dot(data_meaned) / (data_meaned.shape[0] - 1)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    selected_eigenvectors = sorted_eigenvectors[:, :n_components]

    return selected_eigenvectors, sorted_eigenvalues[:n_components]

def pca_transform(data, n_components):
    eigenvectors, _ = compute_pca(data, n_components)
    transformed_data = np.dot(data - np.mean(data, axis=0), eigenvectors)
    return transformed_data
