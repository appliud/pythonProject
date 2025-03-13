import  numpy as np

#seed=1设置为常数，每次运行结果一致
rdm=np.random.RandomState(seed=1)
a=rdm.rand()
b=rdm.rand(2,3)
print("a",a)
print("b",b)