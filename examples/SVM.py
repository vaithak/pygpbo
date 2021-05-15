import pygpbo
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

# Take only the first two features.
data_X = iris.data
data_Y = iris.target

def objectiveSVM(log_C, gamma):
    global data_X, data_Y
    C = np.exp(log_C)

    clf = svm.SVC(C=C, gamma=gamma)
    result  = cross_val_score(clf, data_X, data_Y, scoring='accuracy').mean()
    return result

bounds = {'log_C': (-6, 6), 'gamma': (0.1, 4)}

optimizer = pygpbo.BayesOpt(objectiveSVM, bounds)
optimizer.add_custom_points([{'log_C': 1, 'gamma': 0.1}, 
                             {'log_C': -5, 'gamma': 0.5},
                             {'log_C': 5, 'gamma': 0.5}])

optimizer.maximize(n_iter=30)
print("Optimum point and value: ", optimizer.max_sample)


#### Plotting the surface plot
X = np.linspace(0.1, 4, num=50)
Y = np.linspace(-6 , 6, num=50)
a, b = np.meshgrid(X, Y)

positions = np.vstack([a.ravel(), b.ravel()])
x_test = (np.array(positions)).T
mu_samples = optimizer._gpr.predict(x_test).reshape(a.shape)


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
ax.plot_surface(a, b, mu_samples, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('gamma')
ax.set_ylabel('log(C)')
ax.set_zlabel('SVM accuracy')
ax.set_title("Hyperparameter optimization of SVM for iris dataset")
plt.show()

# Output seen: 
"""
Optimum point and value:  ({'gamma': 0.1, 'log_C': 1.5711129453244772}, array([0.98666667]))
"""
