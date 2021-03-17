import pytest
import numpy as np
from pygpbo import FunctionSpace


bounds = {'b': (-2, 2), 'a': (0, 10), 'c': (-5, 5), 'd': (2, 5)}
# sorted by key names
keys = sorted(bounds)
ret_bounds = np.array([(0.0, 10.0), (-2.0, 2.0), (-5.0, 5.0), (2.0, 5.0)])

def F(b, c, a, d):
    return (a - b - c - d)


class TestFunctionSpace:
    F_space = FunctionSpace(F, bounds, random_state=np.random.RandomState(0))

    def test_bounds(self):
        status = len(self.F_space.bounds) == len(ret_bounds)
        status = status and all((self.F_space.bounds[i][0] == ret_bounds[i][0] and self.F_space.bounds[i][1] == ret_bounds[i][1]) for i in range(len(ret_bounds))) 
        assert status

    def test_dim(self):
        assert self.F_space.dim == len(keys)

    def test_rand(self):
        num = 1
        X, Y = self.F_space.get_rand(number=num, output=True)
        
        status = (X.shape[0] == num and Y.shape[0] == num) and (X.shape[1] == len(ret_bounds))
        for n in range(num):
            for i in range(len(ret_bounds)):
                if (X[n][i] < ret_bounds[i][0]) or (X[n][i] > ret_bounds[i][1]):
                    status = False
                    break

                print(X[n])
                args = dict(zip(keys, X[n]))
                print(args)
                if F(**args) != Y[n]:
                    status = False
                    break

        assert status

