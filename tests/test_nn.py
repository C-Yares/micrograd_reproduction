from micrograd.nn import MLP
model = MLP(3, [4, 4, 1])      # 3 inputs, two hidden layers of 4, 1 output
x = [1.0, -2.0, 3.0]
out = model(x)
print(out)                      # a Value
print(len(model.parameters()))  # 3*4+4 + 4*4+4 + 4*1+1 = 16+4+20+5 wait recompute
# Layer 1: 4 neurons × (3 weights + 1 bias) = 16
# Layer 2: 4 × (4+1) = 20
# Layer 3: 1 × (4+1) = 5
# Total: 41