import numpy as np

class Variable:
	def __init__(self, data):
		self.data = data
		self.grad = None
		self.creater = None

	def set_creater(self, func):
		self.creater = func

	def backward(self):
		funcs = [self.creater]
		while funcs:
			f = funcs.pop()
			x, y = f.input, f.output
			x.grad = f.backward(y.grad)

			if x.creater is not None:
				funcs.append(x.creater)


class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(as_array(y))
		output.set_creater(self)
		self.input = input
		self.output = output
		return output

	def forward(self, x):
		raise NotImplementedError()

	def backward(self, gy):
		raise NotImplementedError()

class Square(Function):
	def forward(self, x):
		y = x ** 2
		return y

	def backward(self, gy):
		x = self.input.data
		return 2 * x * gy

class Exp(Function):
	def forward(self, x):
		return np.exp(x)

	def backward(self, gy):
		x = self.input.data
		return np.exp(x) * gy

def square(x):
	f = Square()
	return f(x)

def exp(x):
	f = Exp()
	return f(x)


class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError('{} is not suppported'.format(type(data)))

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
			x.grad = f.backward(y.grad)
			if x.creater is not None:
				funcs.append(x.creater)


x = Variable(np.array(1.0))
x = Variable(None)

x = np.array(1.0)
y = x ** 2
print(type(x), x.ndim)
print(type(y))

def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x