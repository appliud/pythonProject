import numpy as np
#数组纵向拼接
# 数组
a=np.array([1,2,3])
b=np.array([4,5,6])
c=np.vstack((a,b))
print(c)