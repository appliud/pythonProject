import numpy as np

x,y=np.mgrid[1:3:1,2:4:0.5]
# ravel使x变为一维数组
grid=np.c_[x.ravel(),y.ravel()]
print("x",x)
print("y",y)
print("grid:\n",grid)