# demo_train.py
from micrograd.nn import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])

for step in range(100):
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    for p in model.parameters():
        p.grad = 0.0

    loss.backward()

    lr = 0.03
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 10 == 0:
        print(f"step {step}: loss = {loss.data:.6f}")

print("predictions:", [p.data for p in ypred])
print("targets:    ", ys)