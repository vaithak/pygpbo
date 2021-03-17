import numpy as np
from scipy.stats import norm


def EI(X, X_samples, Y_samples, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_samples = gpr.predict(X_samples)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model, otherwise use np.max(Y_sample).
    mu_samples_opt = np.max(mu_samples)

    with np.errstate(divide='warn'):
        imp = mu - mu_samples_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei.ravel()
