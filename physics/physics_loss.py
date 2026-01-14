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
        )[0]

        #equivalent to:
        # grad_i = torch.autograd.grad(
        #     outputs=q[:, i].sum(),
        #     inputs=t,
        #     create_graph=True,
        #     retain_graph=True,
        # )[0]
        
        # No need to check for None since allow_unused=False will raise an error
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
        )[0]
        
        # No need to check for None since allow_unused=False will raise an error
        qdd_list.append(grad_2_i)
            
    # Create qdd from the list
    qdd = torch.cat(qdd_list, dim=1)
            
    return qdot, qdd

def physics_residual(q, qdot, qdd, parameters, time_period=None):
    """
        Computes the physics residual for the double pendulum dynamics
        given q, qdot, qdd.
        parameters: dict with keys 'm1', 'm2', 'l1', 'l2', 'g'
        time_period: float or None, if time normalization is used
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

    if time_period is not None:
        # Adjust for time normalization (t -> t / T)
        # d/dt = (1/T) d/d(t/T)  =>  d^2/dt^2 = (1/T^2) d^2/d(t/T)^2
        T = time_period
        qdot = qdot / T
        qdd = qdd / (T ** 2)

    M = M_fn(q[:, 0], q[:, 1], m1, m2, l1, l2)  # (N, 2, 2)
    C = C_fn(q[:, 0], q[:, 1], qdot[:, 0], qdot[:, 1], m1, m2, l1, l2, g)  # (N, 2)

    rhs = torch.linalg.solve(M, C.unsqueeze(-1)).squeeze(-1)
    residual = qdd - rhs
        
    return residual