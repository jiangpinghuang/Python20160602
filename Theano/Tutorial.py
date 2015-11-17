import theano.tensor as T
from theano import function

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print f(2, 3)
print f(16.3, 12.1)

from theano import pp
print pp(z)

print z.eval({x : 16.3, y : 12.1})

import theano.tensor
a = theano.tensor.vector()
b = theano.tensor.vector()
out = a + a ** 10
output = a ** 2 + b ** 2 + 2 * a * b
f = theano.function([a], out)
fab = theano.function([a, b], output)
print(f([0, 1, 2]))
print(fab([1, 2], [4, 5]))  # a in [1, 2], b in [4, 5], and a = 1, b = 4, a = 2, b = 5.

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))     # s is sigmoid function.
logistic = theano.function([x], s)
print logistic([[0, 1], [-1, -2]])

s2 = (1 + T.tanh(x / 2)) / 2# s2 is tanh function.
logistic2 = theano.function([x], s2)
print logistic2([[0, 1], [-1, -2]]) 
