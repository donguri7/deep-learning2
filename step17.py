import weakref
import numpy as np

class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				TypeError('{} is not supported'.format(type(data)))

		self.data = data
		self.grad = None
		self.creater = None
		self.generation = 0

	def set_creater(self, func):
		self.creater = func
		self.generation = func.generation + 1

	def cleargrad(self):
		self.grad = None

	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = []
		seen_set = set()

		def add_func(f):
			if f not in seen_set:
				funcs.append(f)
				seen_set.add(f)
				funcs.sort(key=lambda x: x.generation)

		add_func(self.creater)

		while funcs:
			f = funcs.pop()
			gys = [output().grad for output in f.outputs]
			gxs = f.backward(*gys)
			if np.isscalar(gxs):
				gxs = (gxs,)

			for x, gx in zip(f.inputs, gxs):
				if x.grad is None:
					x.grad = gx
				else:
					x.grad = x.grad + gx

				if x.creater is not None:
					add_func(x.creater)

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

		self.generation = max([x.generation for x in inputs])
		for output in outputs:
			output.set_creater(self)
		self.inputs = inputs
		self.outputs = [weakref.ref(output) for output in outputs]
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


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)