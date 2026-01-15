import torch
from physics.physics_loss import physics_residual, compute_derivatives, trajectory_residual, kinetic_residual

def compute_loss(model, batch, parameters_tensor, trajectory_loss_ratio=1.0, time_scale=None, ):
    t, initial_state, state, qdot = batch
    
    # Ensure t is a leaf tensor with proper shape and requires grad
    t = t.detach().view(-1, 1).requires_grad_(True)
    
    # Prepare initial_state for model input
    initial_state = initial_state.detach()  # (batch_size, 4)

    # Forward pass with time and initial state as input
    # Concatenate t and initial_state: (batch_size, 5)
    model_input = torch.cat([t, initial_state], dim=1)
    q_pred = model(model_input)  # (N, 2)
    
    # Verification: ensure gradient graph exists
    if q_pred.grad_fn is None:
        raise RuntimeError("q_pred has no grad_fn. The model is detaching the output!")

    # ---------- Physics loss (all points) ----------
    qdot_pred, qdd_pred = compute_derivatives(q_pred, t)

    physic_res = physics_residual(q_pred, qdot_pred, qdd_pred, parameters_tensor, time_scale=time_scale)
    kinetic_res = kinetic_residual(qdot_pred, qdot)
    trajectory_res = trajectory_residual(q_pred, state)

    kenetic_loss = torch.mean(kinetic_res**2) 
    physics_loss = torch.mean(physic_res**2) 
    trajectory_loss = torch.mean(trajectory_res**2)

    # ---------- Total ----------
    total_loss = (1 - trajectory_loss_ratio) * kenetic_loss + trajectory_loss_ratio * trajectory_loss

    loss_dict = {
        "physics_loss": physics_loss.item(),
        "trajectory_loss": trajectory_loss.item(),
        "kinetic_loss": kenetic_loss.item()
    }

    return total_loss, loss_dict
