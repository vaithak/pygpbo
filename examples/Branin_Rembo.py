# Reference: https://www.sfu.ca/~ssurjano/branin.html

import pygpbo
import numpy as np

# x is an n-dimemsional vector, n>3
def F(x, noise=0):
    results = []
    for i in range(x.shape[0]):
        p1, p2 = x[i][0:2]
        results.append( [-(np.power((p2-5.1/(4*np.power(3.14,2))*np.power(p1,2)+5/3.14*p1-6),2)+10*(1-1/(8*3.14))*np.cos(p1)+10) + noise*np.random.randn() ])

    return np.array(results)

D = 100
d_e = 2
bounds = [(-5, 10), (0, 15)] + [(0, 1) for i in range(D-2)]
n_runs = 5
n_iter = 50

max_val = -1e5
best_X_samples = None
best_Y_samples = None
for run in range(n_runs):
    optimizer = pygpbo.hdbo.REMBO(F, bounds, d_e=d_e)
    optimizer.maximize(n_iter=n_iter)
    print("Run :", run, ", Optimum point and value: ", optimizer.max_sample)
    if max_val < optimizer.max_sample[1][0]:
        max_val = optimizer.max_sample[1][0]
        best_X_samples = optimizer.X_samples
        best_Y_samples = optimizer.Y_samples

print("Maximum value: ", max_val)

# with open(f'REMBO_Branin_{D}_{d_e}_iter{n_iter}.npy', 'wb') as f:
#     np.save(f, best_X_samples)
#     np.save(f, best_Y_samples)
     
# Output seen:
# Run : 0 , Optimum point and value:  (array([ 0.4980707 , -1.41421356]), array([-2.03823053]))
# Run : 1 , Optimum point and value:  (array([ 0.32153678, -0.77836756]), array([-1.02357861]))
# Run : 2 , Optimum point and value:  (array([-1.10414788,  0.55857682]), array([-1.16480553]))
# Run : 3 , Optimum point and value:  (array([-1.12097361, -0.01707791]), array([-0.48507266]))
# Run : 4 , Optimum point and value:  (array([-0.47770884, -1.39775166]), array([-0.47250341]))
# Maximum value:  -0.47250341363395343
