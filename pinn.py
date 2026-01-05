# pinn.py
import torch
import torch.nn as nn
import sympy as sp

from physics import derive_double_pendulum_dynamics

def build_physics_functions():
    """
        Builds numerical functions for the mass matrix M and rest vector C
        of the double pendulum dynamics.
    """
    M_sym, C_sym, sym = derive_double_pendulum_dynamics()

    th1, th2 = sp.symbols("th1 th2")
    th1_d, th2_d = sp.symbols("th1_d th2_d")

    M_sym = M_sym.subs({
        sym["theta1"]: th1,
        sym["theta2"]: th2,
    })

    C_sym = C_sym.subs({
        sym["theta1"]: th1,
        sym["theta2"]: th2,
        sym["theta1_dot"]: th1_d,
        sym["theta2_dot"]: th2_d,
    })

    M_fn = sp.lambdify((th1, th2), M_sym, modules="torch")
    C_fn = sp.lambdify((th1, th2, th1_d, th2_d), C_sym, modules="torch")

    return M_fn, C_fn


M_fn, C_fn = build_physics_functions()

class PINN(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

def physics_residual(model, t):
    q = model(t)                       # (N, 2)
    qdot = torch.autograd.grad(
        q, t, torch.ones_like(q),
        create_graph=True
    )[0]
    qdd = torch.autograd.grad(
        qdot, t, torch.ones_like(qdot),
        create_graph=True
    )[0]

    M = M_fn(q[:, 0], q[:, 1])          # (N, 2, 2)
    C = C_fn(q[:, 0], q[:, 1],
             qdot[:, 0], qdot[:, 1])   # (N, 2)

    res = M @ qdd.unsqueeze(-1) + C.unsqueeze(-1)
    return res.squeeze(-1)



def train():
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t = torch.linspace(0, 10, 500).view(-1, 1)
    t.requires_grad_(True)

    for epoch in range(5000):
        optimizer.zero_grad()

        res = physics_residual(model, t)
        loss = (res**2).mean()

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.3e}")


if __name__ == "__main__":
    train()
