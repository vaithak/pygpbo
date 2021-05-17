# Base class for projection based High Dimensional BO techniques like REMBO, HeSBO

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from pygpbo import utils
import pygpbo.acq.EI

import warnings
INF = 1e8

class Proj_HDBO:
    """
    This class describes ...
    d_e : Dimension of projection
    """
    def __init__(self, F, bounds, d_e, kernel=Matern(nu=2.5), random_state=None, verbose=0, acquisition_fn=pygpbo.acq.EI, gpr_params=None):
        if random_state is None:
            random_state = np.random.RandomState()
        self._rand_state = random_state

        self.F = F
        self.bounds = np.array(bounds)
        self.d_e = d_e
        self.D = self.bounds.shape[0]

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

        # projection matrix, size is D x d_e matrix
        self.A = self.get_proj_matrix()


    def get_proj_matrix(self):
        pass


    # X is an n x D matrix or an n x d_e matrix
    def get_Fvals(self, X):
        if (X.shape[1] == self.D):
            return self.F(self.scale(self.projection_func(X)))
        return self.F(self.scale(self.projection_func(X @ self.A.T)))


    def get_rand(self, number=1, output=True):
        X_points = self._rand_state.uniform(-np.sqrt(self.d_e), np.sqrt(self.d_e), size=(number, self.d_e))
        if not output:
            return X_points

        Y_points = self.get_Fvals(X_points)
        return X_points, Y_points


    def projection_func(self, x):
        x[x<-1] = -1
        x[x>1] = 1
        return x

    # Scaling from the (-1, 1) bounds to the actual bound
    def scale(self, x):
        mu = (self.bounds[:,0] + self.bounds[:,1]) / 2
        diff = (self.bounds[:,1] - self.bounds[:,0]) / 2
        new_x = mu + diff*x
        return new_x


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
      pass


    def maximize(self, n_iter=10):
        if self.X_samples.shape[0] == 0:
            # add a random initial point
            self.X_samples, self.Y_samples = self.get_rand(number=2, output=True)

        for i in range(n_iter):

            # Sklearn's GP throws a large number of warnings at times
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._gpr.fit(self.X_samples, self.Y_samples)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            x_next = self._propose_next_sample()
            
            # Obtain sample from the objective function
            y_next = self.get_Fvals(x_next)

            # Add sample to previous samples
            self.X_samples = np.vstack((self.X_samples, x_next))
            self.Y_samples = np.vstack((self.Y_samples, y_next))


    @property
    def samples(self):
        return self.X_samples, self.Y_samples

    @property
    def max_sample(self):
        X, Y = self.samples
        idx = np.argmax(Y)
        return X[idx], Y[idx]

        


