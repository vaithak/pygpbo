# TODO: Create a class for ALEBO method for High Dimensional Bayesian Optimization
# Reference: https://arxiv.org/abs/2001.11659
def gen(n,d,B):
  results=[]
  while len(results)<n:
    x=np.random.rand(d,1)
    B_inv=np.linalg.pinv(B)

    if (all(B_inv@x)<=1 and all(B_inv@x)>=-1):
      results.append(x)  
  return (np.array(results).T).reshape(n,d)
