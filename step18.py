import weakref
import numpy as np

class Variable:
	def __init__(self, data):
		if data is not None:
			if isinstance(data, np.ndarray):
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

	def backward(self, retain_grad=False):
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
			if not isinstance(gxs, tuple):
				gxs = (gxs,)

			for x, gx in zip(f.inputs, gxs):
				if x.grad is None:
					x.grad = gx
				else:
					x.grad = x.grad + gx

				if x.creater is not None:
					add_func(x.creater)

			if not retain_grad:
				for y in f.outputs:
					y().grad = None 


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

		if Config.enable_backprop:
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

class Config:
	enable_backprop = True


import contextlib

@contextlib.contextmanager
def using_config(name, value):
	old_value = getattr(Config, name)
	setattr(Config, name, value)
	try:
		yield
	finally:
		setattr(Config, name, old_value)

def no_grad():
	return using_config('enable_backprop', False)

with no_grad():
	x = Variable(np.array(2.0))
	y = square(x)