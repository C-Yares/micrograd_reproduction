import torch
from micrograd.engine import Value

def test_basic_ops():
    # Build the same expression in micrograd and PyTorch
    x_mg = Value(-4.0)
    z_mg = 2 * x_mg + 2 + x_mg
    q_mg = z_mg.relu() + z_mg * x_mg
    h_mg = (z_mg * z_mg).relu()
    y_mg = h_mg + q_mg + q_mg * x_mg
    y_mg.backward()

    x_pt = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    z_pt = 2 * x_pt + 2 + x_pt
    q_pt = z_pt.relu() + z_pt * x_pt
    h_pt = (z_pt * z_pt).relu()
    y_pt = h_pt + q_pt + q_pt * x_pt
    y_pt.backward()

    # Forward values match
    assert abs(y_mg.data - y_pt.item()) < 1e-6, f"forward mismatch"

    # Gradients match
    assert abs(x_mg.grad - x_pt.grad.item()) < 1e-6, f"backward mismatch"

    print("PASS: forward and backward match PyTorch")

if __name__ == "__main__":
    test_basic_ops()