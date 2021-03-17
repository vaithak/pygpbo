import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize

from pygpbo import utils
from .FunctionSpace import FunctionSpace
import pygpbo.acq.EI

import warnings
INF = 1e8

class BayesOpt:
    """
    This class describes ...
    """
    def __init__(self, F, bounds, kernel=Matern(nu=2.5), random_state=None, verbose=0, acquisition_fn=pygpbo.acq.EI, gpr_params=None):
        if random_state is None:
            random_state = np.random.RandomState()
        self._rand_state = random_state

        # Create the function space for given bounds and fxn
        self._fxn_space = FunctionSpace(F, bounds, random_state)

        # Custom kernels can also be used
        # Initiliazing Gaussian Process Regression for surrogate model
        gpr_params = utils.get_gpr_params(gpr_params)
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self._rand_state,
            **gpr_params
        )

        # Set acquistion function for deciding the next point
        self._acquisition_fn = acquisition_fn

        # X_samples and Y_samples
        self.X_samples = np.array([])
        self.Y_samples = np.array([])


    """
    X_dict_points is an array of dict, with dict's key = parameter name, value = value for that param
    """
    def add_custom_points(self, X_dict_points):
        status, X_points = self._fxn_space.points_dict_to_arr(X_dict_points)
        if not status:
            raise ValueError("Invalid points entered")

        if self.X_samples.shape[0] == 0:
            self.X_samples = X_points
            self.Y_samples = self._fxn_space.get_Fvals(self.X_samples)
        else:
            Y_vals = self._fxn_space.get_Fvals(X_points)
            self.X_samples = np.vstack((self.X_samples, X_points))
            self.Y_samples = np.vstack((self.Y_samples, Y_vals))


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
      initial_points = self._fxn_space.get_rand(number=n_restarts, output=False)
      min_val = INF
      min_point = None

      for point in initial_points:
        res = minimize(lambda x: -self._acquisition_fn(x.reshape(1, -1), self.X_samples, self.Y_samples, self._gpr), x0=point, bounds=self._fxn_space.bounds, method='L-BFGS-B')
        if res.fun[0] < min_val:
          min_val = res.fun[0]
          min_point = res.x
          
      return min_point.reshape(1, -1)


    def maximize(self, n_iter=10):
        if self.X_samples.shape[0] == 0:
            # add a random initial point
            self.X_samples, self.Y_samples = self._fxn_space.get_rand(number=1, output=True)

        for i in range(n_iter):
            # Sklearn's GP throws a large number of warnings at times
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gpr.fit(self.X_samples, self.Y_samples)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            x_next = self._propose_next_sample()
            
            # Obtain sample from the objective function
            y_next = self._fxn_space.get_Fvals(x_next)

            # Add sample to previous samples
            self.X_samples = np.vstack((self.X_samples, x_next))
            self.Y_samples = np.vstack((self.Y_samples, y_next))


    @property
    def samples(self):
        return self._fxn_space.points_arr_to_dict(self.X_samples), self.Y_samples

    @property
    def max_sample(self):
        X, Y = self.samples
        idx = np.argmax(Y)
        return X[idx], Y[idx]

        
