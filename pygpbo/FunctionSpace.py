class FunctionSpace:
    """
    docstring for FunctionSpace
    """
    def __init__(self, fxn, bounds, random_state):
        self._rand_state = random_state
        self._fxn = fxn
        self._bounds = bounds
        