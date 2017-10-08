import numpy as np

x = np.arange(7*7).reshape(7,7)

stacked = np.stack((x,x), axis = 2)
print(stacked)
print(stacked.shape)
stacked_twice = np.concatenate((stacked, x.reshape(7,7,1)), axis = 2)
print(stacked_twice)
print(stacked_twice.shape)
