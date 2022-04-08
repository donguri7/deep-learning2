if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def f(x):
	return x ** 4 - 2 * x ** 2

def gx2(x):
	return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
	print(x)

	y = f(x)
	x.clear_grad()
	y.backward()

	x.data -= x.grad / gx2(x.data)