from micrograd.engine import Value
# x = Value(3.0)
# print(x)            # Value(data=3.0, grad=0.0)
# print(x.data)       # 3.0
# print(x.grad)       # 0.0

# x = Value(2.0)
# y = Value(3.0)
# z = x + y
# print(z)              # Value(data=5.0, grad=0.0)
# print(z._prev)        # {Value(data=2.0...), Value(data=3.0...)}
# print(z._op)          # '+'

# x = Value(2.0)
# y = Value(3.0)
# z = x * y
# print(z)              # Value(data=6.0, grad=0.0)
# print(z._prev)        # the two parents

# x = Value(2.0)
# y = Value(3.0)
# z = x + y
# z.grad = 1.0
# z._backward()
# print(x.grad, y.grad)  # 1.0 1.0   (both get 1*1)

# # Reset
# x = Value(2.0); y = Value(3.0)
# z = x * y
# z.grad = 1.0
# z._backward()
# print(x.grad, y.grad)  # 3.0 2.0   (x.grad gets y.data, y.grad gets x.data)

# x = Value(2.0)
# y = Value(3.0)
# z = x * y + x       # z = 6 + 2 = 8

# z.backward()
# # z = xy + x
# # dz/dx = y + 1 = 4
# # dz/dy = x = 2
# print(x.grad, y.grad)  # 4.0 2.0

# x = Value(3.0)
# y = Value(4.0)
# z = x ** 2 + y / 2
# z.backward()
# # z = x^2 + y/2 = 9 + 2 = 11
# # dz/dx = 2x = 6
# # dz/dy = 1/2
# print(z.data, x.grad, y.grad)  # 11.0 6.0 0.5

# x = Value(-2.0); y = x.relu(); y.backward()
# print(y.data, x.grad)  # 0.0 0.0   (relu kills negatives)

# x = Value(3.0); y = x.relu(); y.backward()
# print(y.data, x.grad)  # 3.0 1.0

