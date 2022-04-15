if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np 
import dezero
from dezero.core import Function
from dezero import utils

class Sin(Function):
	def forward(self, x):
		y = np.sin(x)
		return y

	def backward(self, gy):
		x, = self.inputs
		gx = gy * cos(x)
		return gx

def sin(x):
	return Sin()(x)

class Cos(Function):
	def forward(self, x):
		y = np.cos(x)
		return y

	def backward(self, gy):
		x, = self.inputs
		gx = gy * -sin(x)
		return gx

def cos(x):
	return Cos()(x)

class Tanh(Function):
	def forward(self, x):
		y = np.tanh(x)
		return y

	def backward(self, gy):
		y = self.outputs[0]()
		gx = gy * (1 - y ** 2)
		return gx

def tanh(x):
	return Tanh()(x)

class Exp(Function):
	def forward(self, x):
		y = np.exp(x)
		return y

	def backward(self, gy):
		y = self.outputs[0]()
		gx = gy * y
		return gx

def exp(x):
	return Exp()(x)

class Log(Function):
	def forward(self, x):
		y = np.log(x)
		return y

	def backward(self, gy):
		x, = self.inputs
		gx = gy / x
		return gx

def log(x):
	return Log()(x)


class Reshape(Function):
	def __init__(self, shape):
		self.shape = shape

	def forward(self, x):
		self.x_shape = x.shape
		y = x.reshape(self.shape)
		return y

	def backward(self, gy):
		return reshape(gy, self.x_shape)

from dezero.core import as_variable

def reshape(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return Reshape(shape)(x)


class Transpose(Function):
	def forward(self, x):
		y = np.transpose(x)
		return y

	def backward(self, gy):
		gx = transpose(gy)
		return gx

def transpose(x):
	return Transpose()(x)


class Sum(Function):
	def __init__(self, axis, keepdims):
		self.axis = axis
		self.keepdims = keepdims

	def forward(self, x):
		self.x_shape = x.shape
		y = x.sum(axis=self.axis, keepdims=self.keepdims)
		return y

	def backward(self, gy):
		gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, 
			                            self.keepdims)
		gx = broadcast_to(gy, self.x_shape)
		return gx

def sum(x, axis=None, keepdims=False):
	return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
	def __init__(self, shape):
		self.shape = shape

	def forward(self, x):
		self.x_shape = x.shape
		y = np.broadcast_to(x, self.shape)
		return y

	def backward(self, gy):
		gx = sum_to(gy, self.x_shape)
		return gx

def broadcast_to(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return BroadcastTo(shape)(x)


class SumTo(Function):
	def __init__(self, shape):
		self.shape = shape

	def forward(self, x):
		self.x_shape = x.shape
		y = utils.sum_to(x, self.shape)
		return y

	def backward(self, gy):
		gx = broadcast_to(gy, self.x_shape)
		return gx

def sum_to(x, shape):
	if x.shape == shape:
		return as_variable(x)
	return SumTo(shape)(x)


class Matmul(Function):
	def forward(self, x, W):
		y = x.dot(W)
		return y

	def backward(self, gy):
		x, W = self.inputs
		gx = matmul(gy, W.T)
		gW = matmul(x.T, gy)
		return gx, gW

def matmul(x, W):
	return Matmul()(x, W)

class MeanSquaredError(Function):
	def forward(self, x0, x1):
		diff = x0 - x1
		y =  (diff ** 2).sum() / len(diff)
		return y

	def backward(self, gy):
		x0, x1 = self.inputs
		diff = x0 - x1
		gx0 = gy * diff * (2. / len(diff))
		gx1 = -gx0
		return gx0, gx1

def mean_squared_error(x0, x1):
	return MeanSquaredError()(x0, x1)


def linear_simple(x, W, b=None):
	x, W = as_variable(x), as_variable(W)
	t = matmul(x, W)
	if b is None:
		return t

	y = t + b
	t.data = None
	return y

class Linear(Function):
	def forward(self, x, W, b):
		y = x.dot(W)
		if b is not None:
			y += b
		return y

	def backward(self, gy):
		x, W, b = self.inputs
		gb = None if b.data is None else sum_to(gy, b.shape)
		gx = matmul(gy, W.T)
		gW = matmul(x.T, gy)
		return gx, gW, gb

def linear(x, W, b=None):
	return Linear()(x, W, b)

class Sigmoid(Function):
	def forward(self, x):
		y = 1 / (1 + exp(-x))
		return y

	def backward(self, gy):
		y = self.outputs[0]()
		gx = gy * y * (1 - y)
		return gx

def sigmoid(x):
	return Sigmoid()(x)

def sigmoid_simple(x):
	x = as_variable(x)
	y = 1 / (1 + exp(-x))
	return y


class GetItem(Function):
	def __init__(self, slices):
		self.slices = slices

	def forward(self, x):
		y = x[self.slices]
		return y


	def backward(self, gy):
		x, = self.inputs
		f = GetItemGrad(self.slices, x.shape)
		return f(gy)

def get_item(x, slices):
	return GetItem(slices)(x)

class GetItemGrad(Function):
	def __init__(self, slices, in_shape):
		self.slices = slices
		self.in_shape = in_shape

	def forward(self, gy):
		gx = np.zeros(self.in_shape)
		np.add.at(gx, self.slices, gy)
		return gx

	def backward(self, ggx):
		return get_item(ggx, self.slices)

def softmax_simple(x, axis=1):
	x = as_variable(x)
	y = exp(x)
	sum_y = sum(y, axis=axis, keepdims=True)
	return y / sum_y

class Clip(Function):
	def __init__(self, x_min, x_max):
		self.x_min = x_min
		self.x_max = x_max

	def forward(self, x):
		y = np.clip(x, self.x_min, self.x_max)
		return y

	def backward(self, gy):
		x, = self.inputs
		mask = (x.data >= self.x_min) * (x.data <= self.x_max)
		gx = gy * mask
		return gx

def clip(x, x_min, x_max):
	return Clip(x_min, x_max)(x)

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax_simple(x)
        # convert to one-hot
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def softmax_cross_entropy_simple(x, t):
	x, t = as_variable(x), as_variable(t)
	N = x.shape[0]

	p = softmax_simple(x)
	p = clip(p, 1e-15, 1.0)
	log_p = log(p)
	tlog_p = log_p[np.arange(N), t.data]
	y = -1 * sum(tlog_p) / N
	return y