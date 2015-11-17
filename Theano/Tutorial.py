from theano import Param
from theano import function
from theano import pp
from theano import shared
import theano.tensor
import theano.tensor as T
from sympy.strategies.branch.tests.test_core import inc

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print f(2, 3)
print f(16.3, 12.1)

print pp(z)

print z.eval({x : 16.3, y : 12.1})

a = theano.tensor.vector()
b = theano.tensor.vector()
out = a + a ** 10
output = a ** 2 + b ** 2 + 2 * a * b
f = theano.function([a], out)
fab = theano.function([a, b], output)
print(f([0, 1, 2]))
print(fab([1, 2], [4, 5]))  # a in [1, 2], b in [4, 5], and a = 1, b = 4, a = 2, b = 5.

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))  # s is sigmoid function.
logistic = theano.function([x], s)
print logistic([[0, 1], [-1, -2]])

s2 = (1 + T.tanh(x / 2)) / 2  # s2 is tanh function.
logistic2 = theano.function([x], s2)
print logistic2([[0, 1], [-1, -2]]) 

a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff * 2
f = theano.function([a, b], [diff, abs_diff, diff_squared])
print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])  # a and b is matrix 2 * 2, f is 4 * 3 matrix.

x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, Param(y, default=1)], z)
print f(33)
print f(33, 2)

x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)
print f(33)
print f(33, 2)
print f(33, 0, 1)
print f(33, w_by_name=1)
print f(33, w_by_name=1, y=0)

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
print state.get_value()
print accumulator(1)
print state.get_value()
print accumulator(300)
print state.get_value()

state.set_value(-1)
print accumulator(3)
print state.get_value()

decrementor = function([inc], state, updates=[(state, state-inc)])
print decrementor(2)
print state.get_value()

fn_of_state = state * 2 + inc
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1, 3)
print state.get_value()
