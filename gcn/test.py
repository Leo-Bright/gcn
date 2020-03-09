import sys
import numpy as np

m = np.array([[1,1,1,1],
          [2,2,2,2],
          [3,3,3,3],
          [4,4,4,4]])

print(m)
print("======")
m[[2,1],:] = m[[1,2],:]
print(m)