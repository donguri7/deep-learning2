if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

A, B, C, D = 1, 2, 3, 4
x = np.random.randn(A, B, C, D)
y = x.transpose(1, 0, 3, 2)
print(y.shape)