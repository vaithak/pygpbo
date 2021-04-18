# TODO: Create a class for HesBO method for High Dimensional Bayesian Optimization
# Reference: http://proceedings.mlr.press/v97/nayebi19a.html
import random
import numpy as np
def gen_A(m,n):
  # m x n dimension
  A=np.zeros(shape=(m,n))
  for i in range(m):
    rand_index=np.random.randint(low=0,high=n,size=1)
    A[i,rand_index]=random.choice([1,-1])
  return A  
