import torch
from physics.physics_loss import physics_residual, compute_derivatives, data_residual

def compute_loss(model, batch, parameters_tensor, loss_weights, residual_weights, time_scale=None, num_segments=None):
    t, initial_state, q, qdot, segment_idx = batch
    
    # Ensure proper tensor shapes
    t = t.detach().view(-1, 1).requires_grad_(True)  # (N, 1)
    segment_idx = segment_idx.view(-1)  # (N,) - ensure 1D
    initial_state = initial_state.detach()  # (N, 4)

    # FORWARD PASS (batch points)
    model_input = torch.cat([t, initial_state], dim=1)
    q_pred = model(model_input)  # (N, 2)
    
    if q_pred.grad_fn is None:
        raise RuntimeError("q_pred has no grad_fn. The model is detaching the output!")

    qdot_pred, qdd_pred = compute_derivatives(q_pred, t)

    physic_res = physics_residual(q_pred, qdot_pred, qdd_pred, parameters_tensor, time_scale=time_scale)
    physics_loss = torch.mean(physic_res**2 * residual_weights[segment_idx].unsqueeze(-1)) # (N, 2) * (N, 1) broadcast res -> (N, 2)
    
    data_res = data_residual(q_pred, q)
    data_loss = torch.mean(data_res**2)

    # INITIAL CONDITION LOSS
    # Evaluate IC for all points: forward pass at t=0 with their initial states
    t_zero = torch.zeros_like(t, requires_grad=True)
    
    ic_input = torch.cat([t_zero, initial_state], dim=1)
    q_ic_pred = model(ic_input)  # (N, 2)
    
    # Compute derivatives at t=0
    qdot_ic_pred, _ = compute_derivatives(q_ic_pred, t_zero)

    if time_scale is not None:
        qdot_ic_pred = qdot_ic_pred / time_scale
    
    # IC loss: position + velocity at t=0
    ic_position_loss = torch.mean((q_ic_pred - initial_state[:, :2])**2)
    ic_velocity_loss = torch.mean((qdot_ic_pred - initial_state[:, 2:])**2)
    ic_loss = ic_position_loss + ic_velocity_loss

    # TOTAL LOSS
    total_loss = (loss_weights['physics_lambda'] * physics_loss + 
                  loss_weights['data_lambda'] * data_loss + 
                  loss_weights['ic_lambda'] * ic_loss)

    loss_dict = {
        "physics_loss": physics_loss.item(),
        "data_loss": data_loss.item(),
        "ic_loss": ic_loss.item(),
    }

    # ========== SEGMENT LOSSES ==========
    segment_losses = None
    if num_segments is not None:
        squared_res = torch.mean(physic_res**2, dim=1)
        segment_losses = torch.zeros(num_segments, device=physic_res.device)
        segment_counts = torch.zeros(num_segments, device=physic_res.device)
        segment_losses.scatter_add_(0, segment_idx, squared_res)
        segment_counts.scatter_add_(0, segment_idx, torch.ones_like(squared_res))
        segment_losses = segment_losses / (segment_counts + 1e-8)

    return total_loss, loss_dict, segment_losses
