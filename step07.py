import numpy as np

class Variable:
	def __init__(self, data):
		self.data = data
		self.grad = None
		self.creater = None

	def set_creater(self, func):
		self.creater = func


class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(y)
		output.set_creater(self)
		self.input = input
		self.output = output
		return output

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


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creater == C
assert y.creater.input == b
assert y.creater.input.creater == B
assert y.creater.input.creater.input == a
assert y.creater.input.creater.input.creater == A 
assert y.creater.input.creater.input.creater.input == x


y.grad = np.array(1.0)

C = y.creater
b = C.input
b.grad = C.backward(y.grad)
B = b.creater 
a = B.input
a.grad = B.backward(b.grad)
A = a.creater
x = A.input
x.grad = A.backward(a.grad)
#print(x.grad)


class Variable:
	def __init__(self, data):
		self.data = data
		self.grad = None
		self.creater = None

	def set_creater(self, func):
		self.creater = func 

	def backward(self):
		f = self.creater
		if f is not None:
			x = f.input
			x.grad = f.backward(self.grad)
			x.backward()


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)