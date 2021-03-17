import numpy as np

class FunctionSpace:
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added
    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> bounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = FunctionSpace(target_func, bounds, random_state=0)
    >>> x, y = space.get_rand(number=1, output=True)[0]
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, fxn, bounds, random_state):
        self._rand_state = random_state
        self._F = fxn

        # Get the name of the parameters
        self._keys = sorted(bounds)
        self._bounds = np.array(
            [item[1] for item in sorted(bounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )
    

    @property
    def dim(self):
        return len(self._keys)

    @property
    def bounds(self):
        return self._bounds


    def check_bounds(self, point):
         return all( (point[i] >= self._bounds[i][0] and point[i] <= self._bounds[i][1]) for i in range(len(point)) )


    ##
    ## Gets the fvals.
    ##
    ## :param      X:    { input points at which we need to evaluate F }
    ## :type       X:    {ndarray of size (N x d) } 
    ##
    ## :returns:   The Y_vals.
    ## :rtype:     { ndarray of size (N x 1) }
    ##
    def get_Fvals(self, X):
        X_dict_points = self.points_arr_to_dict(X)
        Y_vals = np.array([self._F(**dict_point) for dict_point in X_dict_points])
        return Y_vals.reshape(X.shape[0], -1)


    def get_rand(self, number=1, output=True):
        X_points = self._rand_state.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                       size=(number, self._bounds.shape[0]))
        if not output:
            return X_points

        Y_points = self.get_Fvals(X_points)
        return X_points, Y_points


    def points_arr_to_dict(self, X_points):
        X_dict_points = np.array([dict(zip(self._keys, point)) for point in X_points])
        return X_dict_points


    def points_dict_to_arr(self, X_dict_points):
        X_points = []

        for dict_point in X_dict_points:
            curr_point = []
            if (len(dict_point) != self.dim):
                return False, None

            for key in self._keys:
                if (key not in dict_point):
                    return False, None
                curr_point.append(dict_point[key])

            if not self.check_bounds(curr_point):
                return False, None

            X_points.append(curr_point)

        return True, np.array(X_points)

