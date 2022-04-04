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

	def cleargrad(self):
		self.grad = None

	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = [self.creater]
		while funcs:
			f = funcs.pop()
			gys = [output.grad for output in f.outputs]
			gxs = f.backward(*gys)
			if np.isscalar(gxs):
				gxs = (gxs,)

			for x, gx in zip(f.inputs, gxs):
				if x.grad is None:
					x.grad = gx
				else:
					x.grad = x.grad + gx 

				if x.creater is not None:
					funcs.append(x.creater)


def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x


class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple):
			ys = (ys,)
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

	def backward(self, gy):
		return gy, gy

class Square(Function):
	def forward(self, x):
		return x ** 2

	def backward(self, gy):
		x = self.inputs[0].data
		gx = 2 * x * gy
		return gx

def add(x0, x1):
	return Add()(x0, x1)

def square(x):
	return Square()(x)

x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(x.grad)

x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)