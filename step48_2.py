if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero.datasets

x, t = dezero.datasets.get_spiral(train=True)

import matplotlib.pyplot as plt

print(x[10], t[10])
print(x[110], t[110])
print(len(t))

for i in range(len(t)):
	if t[i] == 0:
		plt.plot(x[i,0], x[i,1], 'o', color='red')
	if t[i] == 1:
		plt.plot(x[i,0], x[i,1], 'x', color='blue')
	if t[i] == 2:
		plt.plot(x[i,0], x[i,1], 'v', color='green')

plt.show()