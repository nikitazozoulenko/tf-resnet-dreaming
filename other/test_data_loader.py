import numpy as np
from data_loader import *

data_loader = data_loader()
data_loader.load_data_arrays()
print(data_loader.labels.shape)
print(data_loader.labels[0])

#shape ska vara (batch_size, 200)




#zeros[range(self.pool_max_arg[i].size), self.pool_max_arg[i]] = delta.ravel()
