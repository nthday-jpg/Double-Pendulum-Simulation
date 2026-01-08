from physics.equations import get_M_fn_torch, get_C_fn_torch
import torch


def compute_derivatives(q, t):
    """
        Computes qdot, qdd for input time t
    """
    qdot = torch.zeros_like(q)
    for i in range(2):
        qdot[:, i] = torch.autograd.grad(
            outputs=q[:, i],
            inputs=t,
            grad_outputs=torch.ones_like(q[:, i]),
            create_graph=True,
            retain_graph=True
        )[0].squeeze()
    
    qdd = torch.zeros_like(q)
    for i in range(2):
        qdd[:, i] = torch.autograd.grad(
            outputs=qdot[:, i],
            inputs=t,
            grad_outputs=torch.ones_like(qdot[:, i]),
            create_graph=True,
            retain_graph=True
        )[0].squeeze()
    return qdot, qdd

def physics_residual(q, qdot, qdd, parameters):
    """
        Computes the physics residual for the double pendulum dynamics
        given q, qdot, qdd.
        parameters: dict with keys 'm1', 'm2', 'l1', 'l2', 'g'
    """
    # Convert parameters to tensors on same device as q
    device = q.device
    m1 = torch.tensor(parameters['m1'], dtype=q.dtype, device=device)
    m2 = torch.tensor(parameters['m2'], dtype=q.dtype, device=device)
    l1 = torch.tensor(parameters['l1'], dtype=q.dtype, device=device)
    l2 = torch.tensor(parameters['l2'], dtype=q.dtype, device=device)
    g = torch.tensor(parameters['g'], dtype=q.dtype, device=device)

    M_fn = get_M_fn_torch()
    C_fn = get_C_fn_torch()
    
    # Type assertion for Pylance
    assert M_fn is not None and C_fn is not None

    M = M_fn(q[:, 0], q[:, 1], m1, m2, l1, l2)  # (N, 2, 2)
    C = C_fn(q[:, 0], q[:, 1], qdot[:, 0], qdot[:, 1], m1, m2, l1, l2, g)  # (N, 2)

    residual = M @ qdd.unsqueeze(-1) + C.unsqueeze(-1)
    residual = residual.squeeze(-1)  # (N, 2)
    return residual