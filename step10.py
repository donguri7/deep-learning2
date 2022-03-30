import unittest
import numpy as np

class Variable:
	def __init__(self, data):
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
		return x ** 2

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

def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x


class SquareTest(unittest.TestCase):
	def test_forward(self):
		x = Variable(np.array(2.0))
		y = square(x)
		expected = np.array(4.0)
		self.assertEqual(y.data, expected)

	def test_backward(self):
		x = Variable(np.array(3.0))
		y = square(x)
		y.backward()
		expected = np.array(6.0)
		self.assertEqual(x.grad, expected)

	def test_gradient_check(self):
		x = Variable(np.random.randn(1))
		y = square(x)
		y.backward()
		num_grad = numerical_diff(square, x)
		flg = np.allclose(x.grad, num_grad)
		self.assertTrue(flg)


def numerical_diff(f, x, eps=1e-4):
	x0 = Variable(x.data - eps)
	x1 = Variable(x.data + eps)
	y0 = f(x0)
	y1 = f(x1)
	return (y1.data - y0.data) / (2 * eps)


