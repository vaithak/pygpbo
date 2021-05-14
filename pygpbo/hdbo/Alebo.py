# TODO: Create a class for ALEBO method for High Dimensional Bayesian Optimization
# Reference: https://arxiv.org/abs/2001.11659
def gen(n,d,B):
  results=np.array([])
  while len(results)<n:
    x=np.random.rand(d,1)
    D = B.shape[1]
    B_inv=np.linalg.pinv(B)
    if ((B_inv@x)<=1).sum()== D and ((B_inv@x)>=-1).sum() == D:
      if results.size==-0:
        results = x.T
      else:
        results = np.concatenate((results,x.T),axis=0)
  return results
