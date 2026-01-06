import torch
from physics.equations import M_fn, C_fn
from physics.physics_loss import physics_residual, compute_derivatives

def compute_loss(model, batch, weight_data=1.0, weight_phys=1.0):
    t, state, point_type = batch
    t = t.view(-1, 1)
    t.requires_grad_(True)

    q_pred = model(t)  # (N, 2)

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
