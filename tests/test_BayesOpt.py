import pytest
import numpy as np
import pygpbo


bounds_1D = {'x': (-4, 4)}
def F_1D(x, noise=0.1):
    # Max value near 10.54, at x near 2.8
    return np.power(x, 2)*np.sin(x) - (3*x*np.cos(x)) + noise*np.random.randn()

bounds_2D = {'a': (4, 8), 'b': (-2, 2)}
def F_2D(b, a, noise=0.1):
    # Inverted parabola, Max value is approx 4, near (a, b) = (6, 2)
    return 4 - (np.power(b, 2) + np.power(a - 6, 2)) + noise*np.random.randn()

class TestBayesOpt:

    def test_1D(self):
        optimizer = pygpbo.BayesOpt(F_1D, bounds_1D)
        optimizer.add_custom_points([
            {'x': 0}, 
            {'x': -2}
        ])
        optimizer.maximize(n_iter=20)
        res_x, res_y = optimizer.max_sample
        print("Optimum point and value: ", res_x, res_y)
        assert (10 <= res_y[0] <= 11) and (2.6 <= res_x['x'] <= 3)

    def test_2D(self):
        optimizer = pygpbo.BayesOpt(F_2D, bounds_2D)
        optimizer.add_custom_points([
            {'a': 4.0, 'b': -2}, 
            {'a': 8.0, 'b': 2}
        ])
        optimizer.maximize(n_iter=20)
        res_x, res_y = optimizer.max_sample
        print("Optimum point and value: ", res_x, res_y)
        assert (3.5 <= res_y[0] <= 4.5) and (5.6 <= res_x['a'] <= 6.4) and (-0.4 <= res_x['b'] <= 0.4)

