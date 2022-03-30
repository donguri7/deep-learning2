import numpy as np 

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
			f = funcs.pop()  # 関数を取得
			x, y = f.input, f.output  # 関数の入出力を取得
			x.grad = f.backward(y.grad)  # backwardメソッドを呼ぶ

			if x.creater is not None:
				funcs.append(x.creater)  # １つ前の関数をリストに追加


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