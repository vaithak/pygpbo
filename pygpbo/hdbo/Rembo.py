# TODO: Create a class for REMBO method for High Dimensional Bayesian Optimization
# Reference: https://arxiv.org/abs/1301.1942

import numpy as np
from .proj_hdbo import Proj_HDBO
import pygpbo.acq.EI
from scipy.optimize import minimize

INF = 1e8

class REMBO(Proj_HDBO):

    def get_proj_matrix(self):
        A = np.random.normal(0,1,(self.D, self.d_e))

        # for hypersphere
        for row in range(self.D):
            norm_const = np.linalg.norm(A[row,:])
            A[row,:] = A[row,:]/norm_const
        return A


    def _propose_next_sample(self, n_restarts=20):
      '''
        Proposes the next sampling point by optimizing the acquisition function.
        
        Args:
            acquisition: Acquisition function.
            X_samples: Sample locations (n x d).
            Y_samples: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.

        Returns:
            Location of the acquisition function maximum.
      '''
      initial_points = self.get_rand(number=n_restarts, output=False)
      min_val = INF
      min_point = None

      bounds = np.zeros((self.d_e, 2))
      bounds[:,0] = -np.sqrt(self.d_e)
      bounds[:,1] = np.sqrt(self.d_e)

      for point in initial_points:
        res = minimize(lambda x: -self._acquisition_fn(x.reshape(1, -1), self.X_samples, self.Y_samples, self._gpr), 
                        x0=point, bounds=bounds, method='L-BFGS-B')
        if res.fun[0] < min_val:
          min_val = res.fun[0]
          min_point = res.x
          
      return min_point.reshape(1, -1)

