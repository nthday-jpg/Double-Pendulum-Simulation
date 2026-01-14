from physics.equations import get_M_fn_torch, get_C_fn_torch
import torch


def compute_derivatives(q, t):
    """
    Computes qdot (dq/dt) and qdd (d^2q/dt^2) for input time t.
    """
    # 1. Compute qdot (First Derivative)
    qdot_list = []
    for i in range(2):
        grad_i = torch.autograd.grad(
            outputs=q[:, i],
            inputs=t,
            grad_outputs=torch.ones_like(q[:, i]),
            create_graph=True,   # Essential for computing 2nd derivative
            retain_graph=True,   # Keeps graph alive for the next loop iteration
            allow_unused=False
        )[0]
        
        if grad_i is None:
            # Handle cases where a coordinate might not depend on t
            qdot_list.append(torch.zeros_like(t))
        else:
            qdot_list.append(grad_i)
    
    # Create qdot from the list (preserves the graph history)
    qdot = torch.cat(qdot_list, dim=1) 

    # 2. Compute qdd (Second Derivative)
    qdd_list = []
    for i in range(2):
        grad_2_i = torch.autograd.grad(
            outputs=qdot[:, i],
            inputs=t,
            grad_outputs=torch.ones_like(qdot[:, i]),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        
        if grad_2_i is None:
            qdd_list.append(torch.zeros_like(t))
        else:
            qdd_list.append(grad_2_i)
            
    # Create qdd from the list
    qdd = torch.cat(qdd_list, dim=1)
            
    return qdot, qdd

def physics_residual(q, qdot, qdd, parameters, normalize=True):
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

    if normalize:
        scale_0 = (m1 + m2) * l1 * l1 * g
        scale_1 = m2 * l2 * l2 * g
        
        scale_0 = torch.clamp(scale_0, min=1e-8)
        scale_1 = torch.clamp(scale_1, min=1e-8)
        
        residual[:, 0] = residual[:, 0] / scale_0
        residual[:, 1] = residual[:, 1] / scale_1
        
    return residual