import numpy as np
from scipy.special import psi, polygamma

def dirichlet_moments(P):
    """
    Estimate Dirichlet parameters using method of moments.
    
    Parameters:
        P: (N, V) matrix, each row a multinomial (sums to 1)
        
    Returns:
        alpha: estimated Dirichlet parameter vector
    
    Implementation based on the formula:
        alpha_i = E[X_i] * ((E[X_j](1-E[X_j])/V[X_j]) - 1)
        where j is any index, possibly i itself
    """
    N, V = P.shape
    
    # Ensure P is valid (all entries positive and rows sum to 1)
    P = np.maximum(P, 1e-8)  # Prevent division by zero
    # P = P / np.sum(P, axis=1, keepdims=True)  # Normalize rows
    
    # Calculate E[X_i] for each dimension (mean)
    E_X = np.mean(P, axis=0)
    
    # Calculate V[X_i] for each dimension (variance)
    V_X = np.var(P, axis=0)
    
    # Prevent division by zero in variance
    V_X = np.maximum(V_X, 1e-8)
    
    # Apply the method of moments formula directly:
    # α_i = E[X_i] * ((E[X_i](1-E[X_i])/V[X_i]) - 1)
    # We use i=j as given in the formula description
    precision = np.zeros(V)
    
    for i in range(V):
        # Calculate the precision factor
        precision_factor = (E_X[i] * (1 - E_X[i]) / V_X[i]) - 1
        
        # Ensure the precision factor is positive
        precision_factor = max(precision_factor, 1e-7)
        
        # Calculate alpha_i
        precision[i] = precision_factor
    
    # The final alpha values
    alpha = E_X * precision
    
    # Ensure all alphas are positive
    alpha = np.maximum(alpha, 0.001)
    
    return alpha

def dirichlet_minka_fixed_point(P, tol=1e-6, max_iter=1000):
    """
    Estimate Dirichlet parameters using Minka's fixed-point iteration (2000).
    P: (N, V) matrix, each row a multinomial (sums to 1)
    Returns: alpha vector
    """
    N, V = P.shape
    log_p = np.log(P)
    mean_log_p = np.mean(log_p, axis=0)
    # Initialize alpha
    alpha = np.ones(V)
    for _ in range(max_iter):
        alpha0 = np.sum(alpha)
        grad = N * (psi(alpha0) - psi(alpha) + mean_log_p)
        hess = -N * polygamma(1, alpha)
        z = N * polygamma(1, alpha0)
        c = np.sum(grad / hess) / (1.0 / z + np.sum(1.0 / hess))
        update = (grad - c) / hess
        alpha_new = alpha - update
        # Project to positive values
        alpha_new = np.maximum(alpha_new, 1e-6)
        if np.all(np.abs(alpha_new - alpha) < tol):
            break
        alpha = alpha_new
    return alpha

def scale_alpha(alpha, a=0.01, b=1.0):
    """
    Scale alpha to be in range a and b.
    min-max scaling to [a, b].
    
    returns:
        scaled_alpha: scaled alpha vector
    """
    return a + (alpha - alpha.min()) * (b - a) / (alpha.max() - alpha.min())
    
    
# Example usage:
if __name__ == "__main__":
    import sys
    is_synthetic = 0  # Set to False to load your own data
    # Generate synthetic data
    if is_synthetic:
        np.random.seed(42)
        true_alpha = np.array([2.0, 5.0, 1.0, 0.5, 3.0, 0.01, 0.2, 0.07, 0.01, 0.01])
        print("True alpha:", true_alpha)

        n_samples = 1000
        # Generate samples from Dirichlet distribution
        samples = np.random.dirichlet(true_alpha, size=n_samples)
    
    else:
        # load your data here
        import torch
        import sys
        if len(sys.argv) > 1:
            dataset = sys.argv[1]
            metric = sys.argv[2]
            num_topics = sys.argv[3]
        samples = torch.load(f"priors/{dataset}/{num_topics}/{metric}_prior.pt").T
        
    # Estimate parameters using method of moments or Minka's method
    use_minka = False
    if len(sys.argv) > 4:
        use_minka = bool(int(sys.argv[4]))
    if use_minka:
        estimated_alpha = dirichlet_minka_fixed_point(samples)
        print("Estimated alpha (Minka fixed-point):", estimated_alpha)
    else:
        estimated_alpha = dirichlet_moments(samples)
        print("Estimated alpha (method of moments):", estimated_alpha)
    # a = 1e-6
    # b = 0.1
    # scaled_a = scale_alpha(estimated_alpha, a=a, b=b)
    import os
    path = f"priors/{dataset}/{num_topics}"
    if not os.path.exists(path):
        os.makedirs(path)
    
    if use_minka:
        np.savetxt(f"{path}/minka_{metric}.csv", [estimated_alpha], delimiter=",", fmt="%.6f")
    else:
        np.savetxt(f"{path}/mom_{metric}.csv", [estimated_alpha], delimiter=",", fmt="%.6f")
        print(f"Saved estimated alpha to {path}/mom_{metric}.csv")