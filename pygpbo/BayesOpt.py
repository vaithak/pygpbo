import utils
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from acq_fn import EI
import FunctionSpace

# TODO: Import default acquisition fn

class BayesOpt:
    """
    This class describes ...
    """
    def __init__(self, F, bounds, kernel=None, random_state=None, verbose=0, acquisition_fn=None, gpr_params=None):
        if random_state is None:
            random_state = np.random.RandomState()
        self._rand_state = random_state

        # Create the function space for given bounds and fxn
        self._fxn_space = FunctionSpace(F, bounds, random_state)

        # Custom kernels can also be used
        if kernel is None:
            kernel = Matern(nu=2.5)

        # Initiliazing Gaussian Process Regression for surrogate model
        gpr_params = utils.get_gpr_params(gpr_params)
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=self._random_state,
            **gpr_params
        )

        # Set acquistion function for deciding the next point
        if acquisition_fn is None:
            acquisition_fn = EI
        self._acquisition_fn = acquisition_fn


    def maximize(self, n_iter=20):

        
