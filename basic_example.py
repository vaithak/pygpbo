import numpy as np
import pygpbo

def F(x, noise=0.2):
    return (np.power(x, 2))*np.sin(x) - (3*x*np.cos(x)) + noise*np.random.randn()

bounds = {'x': (-4, 4)}

optimizer = pygpbo.BayesOpt(F, bounds)
optimizer.add_custom_points([
    {'x': 0}, 
    {'x': -2}
])

optimizer.maximize()
print("Optimum point and value: ", optimizer.max_sample)