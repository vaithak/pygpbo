import numpy as np

n_iter = 10000
D_arr = [20, 100, 500]
d_e_arr = [1, 2, 3, 4, 5]
results = np.zeros((len(D_arr), len(d_e_arr)))

for i in range(len(D_arr)):
    for j in range(len(d_e_arr)):
        D, d_e = D_arr[i], d_e_arr[j]

        A = np.random.randn(D, d_e)
        X = np.random.uniform(-np.sqrt(d_e), np.sqrt(d_e), (n_iter, d_e))
        X_proj = X @ A.T

        count = 0.0
        for it in range(n_iter):
            if (X_proj[it]<=1).all() and (X_proj[it]>=-1).all():
                count += 1.0
        results[i, j] = count / n_iter



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,5))
for i in range(len(D_arr)):
    plt.plot(d_e_arr, results[i], label=f'D = {D_arr[i]}', marker='x')

plt.legend(loc='best')
plt.xlabel("Embedding dimension d_e")
plt.xticks(d_e_arr)
plt.ylabel("Probability that projection\n satisfies box bounds")
plt.show()

