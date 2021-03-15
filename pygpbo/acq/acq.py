# TODO: Implement the base class for all acquisition functions
import scipy.optimize as optimizer

class acq:
    """docstring for acq"""
    def __init__(self, max_method):
        self._max_method = max_method
        