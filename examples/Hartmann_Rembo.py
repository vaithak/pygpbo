import pygpbo
import numpy as np

# X is an n-dimemsional vector, n>6
def F(X, noise=0):
    results = []
    for i in range(X.shape[0]):
        x1, x2, x3, x4, x5, x6 = X[i][0:6]
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

        results.append([-res])  # - for minimize

    return np.array(results)

D = 100
d_e = 6
bounds = [(0, 1) for i in range(D)]
n_runs = 10
n_iter = 200

final_avg_res = np.zeros((n_iter + 2, 1))
run = 0
while (run < n_runs):
    # try:
        optimizer = pygpbo.hdbo.REMBO(F, bounds, d_e=d_e)
        optimizer.maximize(n_iter=n_iter)
        print("Run: ", run, ", Optimum point and value: ", optimizer.max_sample[1])
        prefix_max = np.maximum.accumulate(optimizer.Y_samples)
        final_avg_res += prefix_max
        run += 1
    # except:
        # print("Run: ", run, ", Exception")

final_avg_res /= (1.0*n_runs)
np.save("hartmann_100.npy", final_avg_res)
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
