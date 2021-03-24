import numpy as np
from scipy.stats import norm


def PI(X,X_t,gpr,e):
    '''
    Computes the PI at points X based on existing samples X_t
    using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_t: Sample locations (n x d).
        gpr: A GaussianProcessRegressor fitted to samples.
        e: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''

  y_t = gpr.predict(X_t)
  X= np.expand_dims(X,axis=0)
  y,std = gpr.predict(X,return_std=True)
  
  std = std.reshape(-1,1)
  best_y = np.max(y_t)
  return norm.cdf((y-best_y-e)/std) 