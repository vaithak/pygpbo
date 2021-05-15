# Reference: https://www.sfu.ca/~ssurjano/hart6.html

import pygpbo
import numpy as np


def F(x1, x2, x3, x4, x5, x6, noise=0):
    x = np.array([x1, x2, x3, x4, x5, x6])
    
    A = np.array([[10  , 3  , 17  , 3.50, 1.7, 8 ],
                  [0.05, 10 , 17  , 0.1 , 8  , 14],
                  [3   , 3.5, 1.7 , 10  , 17 , 8 ],
                  [17  , 8  , 0.05, 10  , 0.1, 14]])

    P = 1e-4 * np.array([[1312, 1696, 5569, 124 , 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    res = 0.0
    for i in range(4):
        curr_res = 0.0
        for j in range(6):
            curr_res += A[i, j] * ((x[j] - P[i, j])**2)
        res -= (alpha[i] * np.exp(-curr_res))

    return -res  # - for minimize


bounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1), 'x6': (0, 1)}

optimizer = pygpbo.BayesOpt(F, bounds)
optimizer.add_custom_points([{'x1': 0.5, 'x2': 0.5, 'x3': 0.5, 'x4': 0.5, 'x5': 0.5, 'x6': 0.5},
                             {'x1': 0.6, 'x2': 0.6, 'x3': 0.1, 'x4': 0.1, 'x5': 0.1, 'x6': 0.1},
                             {'x1': 0.1, 'x2': 0.5, 'x3': 0.5, 'x4': 0.5, 'x5': 0.5, 'x6': 0.9}])

optimizer.maximize(n_iter=150)
print("Optimum point and value: ", optimizer.max_sample)

# Output seen: 
"""
Optimum point and value:  ({'x1': 0.19152669773147596, 'x2': 0.15664682129489924, 'x3': 0.5080429885461548, 
                            'x4': 0.27836559113430875, 'x5': 0.3130313033982394, 'x6': 0.6465736438253549}, 
                            array([3.30806842]))
"""

