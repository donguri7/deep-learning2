import numpy as np 

class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				TypeError('{} is not supported'.format(type(data)))

		self.data = data
		self.grad = None
		self.creater = None 

	def set_creater(self, func):
		self.creater = func 

	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = [self.creater]
		while funcs:
			f = funcs.pop()
			x, y = f.input, f.output
			x.gad = f.backward(y.grad)
			if x.creater is not None:
				funcs.append(x.creater)


def as_array(x):
	if  np.isscalar(x):
		return np.array(x)
	return x


class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple):
			ys = (ys, )
		outputs = [Variable(as_array(y)) for y in ys]

		for output in outputs:
			output.set_creater(self)
		self.inputs = inputs
		self.outputs = outputs
		return outputs if len(outputs) > 1 else outputs[0]

	def forward(self, xs): 
		raise NotImplementedError()

	def backward(self, gys):
		raise NotImplementedError()


class Add(Function):
	def forward(self, x0, x1):
		y = x0 + x1
		return y


def add(x0, x1):
	return Add()(x0, x1)


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)