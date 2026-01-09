import torch
from physics.physics_loss import physics_residual, compute_derivatives

def compute_loss(model, batch, weight_data=1.0, weight_phys=1.0):
    t, initial_state, state, point_type = batch
    
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

    parameters = {
        'm1': 1.0,
        'm2': 1.0,
        'l1': 1.0,
        'l2': 1.0,
        'g': 9.81
    }

    residual = physics_residual(q_pred, qdot_pred, qdd_pred, parameters)
    physics_loss = torch.mean(residual**2)

    # ---------- Data loss (data points only) ----------
    data_mask = (point_type == 0)

    if torch.any(data_mask):
        q_data = state[data_mask]
        q_pred_data = q_pred[data_mask]
        data_loss = torch.mean((q_pred_data - q_data)**2)
    else:
        data_loss = torch.tensor(0.0, device=t.device)

    # ---------- Total ----------
    total_loss = weight_phys * physics_loss + weight_data * data_loss

    loss_dict = {
        "physics_loss": physics_loss.item(),
        "data_loss": data_loss.item()
    }

    return total_loss, loss_dict
