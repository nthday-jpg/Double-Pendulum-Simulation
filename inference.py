import torch
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from models.pinn import PINN
from physics.equations import double_pendulum_derivatives, compute_energy

"""
    # Command line
    python interence.py runs/my_run/checkpoints/best_model.pth --output-dir results --t-end 20 --num-points 2000

    # With specific initial conditions
    python interence.py checkpoint.pth --theta1 0.5 --theta2 -0.3 --omega1 0.1 --omega2 -0.2
"""


def load_model(checkpoint_path, device='cpu'):
    """Load trained PINN model from checkpoint with normalization parameters."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    from utils.config import Config
    cfg = Config(**checkpoint['config'])
    model = PINN(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract normalization parameters from checkpoint
    normalization_params = {
        'normalize_time': checkpoint.get('normalize_time', getattr(cfg, 'normalize_time', True)),
        'normalize_angles': checkpoint.get('normalize_angles', getattr(cfg, 'normalize_angles', True)),
        't_min': checkpoint.get('t_min', 0.0),
        't_max': checkpoint.get('t_max', 1.0),
        'theta_min': checkpoint.get('theta_min', -np.pi),
        'theta_max': checkpoint.get('theta_max', np.pi),
    }
    
    return model, cfg, normalization_params


def simulate_with_pinn(model, initial_state, t_span, num_points, 
                       normalize_time=True, normalize_angles=True,
                       t_min=0.0, t_max=1.0, 
                       theta_min=-np.pi, theta_max=np.pi,
                       device='cpu'):
    """
    Simulate double pendulum trajectory using trained PINN model.
    Applies same normalization as training dataset.
    
    Args:
        model: Trained PINN model
        initial_state: [theta1_0, theta2_0, omega1_0, omega2_0] in PHYSICAL units
        t_span: (t_start, t_end) in PHYSICAL units
        num_points: Number of time points
        normalize_time: Whether time was normalized during training
        normalize_angles: Whether angles were normalized during training
        t_min, t_max: Global time normalization params from training dataset
        theta_min, theta_max: Angle normalization params (default: [-π, π])
        device: Device to run on
        
    Returns:
        t: Time array (num_points,) in PHYSICAL units
        q: Position array (num_points, 2) = [theta1, theta2] in PHYSICAL units
        qdot: Velocity array (num_points, 2) = [omega1, omega2] in PHYSICAL units
    """
    model.eval()
    
    # Create time points in physical units
    t = np.linspace(t_span[0], t_span[1], num_points)
    
    # Normalize time if needed (SAME AS DATASET!)
    if normalize_time:
        t_norm = (t - t_min) / (t_max - t_min)  # → [0, 1]
        t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)
    else:
        t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Normalize initial state if needed (SAME AS DATASET!)
    initial_state_copy = np.array(initial_state, dtype=np.float32).copy()
    if normalize_angles:
        # Normalize angles: [-π, π] → [0, 1] → [-1, 1]
        initial_state_copy[:2] = (initial_state_copy[:2] - theta_min) / (theta_max - theta_min)
        initial_state_copy[:2] = 2 * initial_state_copy[:2] - 1  # [0,1] → [-1,1]
        # Note: omegas are kept as-is for now (TODO: add omega normalization if dataset uses it)
    
    # Prepare initial state tensor - repeat for all time points
    initial_state_tensor = torch.tensor(initial_state_copy, dtype=torch.float32).unsqueeze(0).to(device)
    initial_state_batch = initial_state_tensor.repeat(num_points, 1)
    
    # Concatenate time and initial state: [t, theta1_0, theta2_0, omega1_0, omega2_0]
    x = torch.cat([t_tensor, initial_state_batch], dim=1)
    
    # Get predictions from model (in normalized space)
    with torch.no_grad():
        predictions = model(x).cpu().numpy()
    
    # Denormalize predictions to physical units
    if predictions.shape[1] == 2:
        q_norm = predictions  # (num_points, 2)
        
        if normalize_angles:
            # Denormalize: [-1, 1] → [0, 1] → [-π, π]
            q = (q_norm + 1) / 2  # [-1,1] → [0,1]
            q = q * (theta_max - theta_min) + theta_min  # [0,1] → [-π, π]
        else:
            q = q_norm
        
        # Compute velocities from denormalized positions using PHYSICAL time
        qdot = np.gradient(q, t, axis=0)
        
    elif predictions.shape[1] >= 4:
        # If model outputs [theta1, theta2, omega1, omega2]
        q_norm = predictions[:, :2]
        qdot_norm = predictions[:, 2:4]
        
        if normalize_angles:
            # Denormalize positions
            q = (q_norm + 1) / 2
            q = q * (theta_max - theta_min) + theta_min
            # TODO: Denormalize velocities if they were normalized
            qdot = qdot_norm
        else:
            q = q_norm
            qdot = qdot_norm
    else:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    
    return t, q, qdot


def simulate_ground_truth(initial_state, t_span, num_points, 
                         m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """
    Simulate ground truth trajectory using numerical integration.
    
    Args:
        initial_state: [theta1_0, theta2_0, omega1_0, omega2_0]
        t_span: (t_start, t_end)
        num_points: Number of time points
        m1, m2: Rod masses
        l1, l2: Rod lengths
        g: Gravity
        
    Returns:
        t: Time array (num_points,)
        q: Position array (num_points, 2) = [theta1, theta2]
        qdot: Velocity array (num_points, 2) = [omega1, omega2]
    """
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    solution = solve_ivp(
        double_pendulum_derivatives,
        t_span,
        initial_state,
        args=((m1, m2, l1, l2, g),),
        t_eval=t_eval,
        method='DOP853',
        rtol=1e-8,
        atol=1e-10
    )
    
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    
    t = solution.t
    q = solution.y[:2, :].T  # [theta1, theta2]
    qdot = solution.y[2:, :].T  # [omega1, omega2]
    
    return t, q, qdot


def save_trajectory(output_dir, t, q, qdot, prefix="trajectory", parameters=None):
    """
    Save trajectory in the same format as generator.
    
    Args:
        output_dir: Directory to save results
        t: Time array
        q: Position array
        qdot: Velocity array
        prefix: Filename prefix
        parameters: Physical parameters dict (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trajectory
    traj_filename = os.path.join(output_dir, f"{prefix}.npz")
    np.savez(traj_filename, t=t, q=q, qdot=qdot)
    print(f"Saved trajectory: {traj_filename}")
    
    # Save parameters if provided
    if parameters is not None:
        params_filename = os.path.join(output_dir, f"{prefix}_parameters.json")
        with open(params_filename, "w") as f:
            json.dump(parameters, f, indent=2)
        print(f"Saved parameters: {params_filename}")


def plot_comparison(t_true, q_true, qdot_true, t_pred, q_pred, qdot_pred, output_dir):
    """
    Plot comparison between ground truth and prediction.
    Creates 4 subplots: theta1, theta2, omega1, omega2.
    
    Args:
        t_true: Time array for ground truth
        q_true: Position array (N, 2) for ground truth [theta1, theta2]
        qdot_true: Velocity array (N, 2) for ground truth [omega1, omega2]
        t_pred: Time array for predictions
        q_pred: Position array (N, 2) for predictions [theta1, theta2]
        qdot_pred: Velocity array (N, 2) for predictions [omega1, omega2]
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PINN Predictions vs Ground Truth', fontsize=16, fontweight='bold')
    
    # Plot theta1
    axes[0, 0].plot(t_true, q_true[:, 0], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[0, 0].plot(t_pred, q_pred[:, 0], 'r--', linewidth=2, label='PINN Prediction', alpha=0.8)
    axes[0, 0].set_xlabel('Time (s)', fontsize=12)
    axes[0, 0].set_ylabel('θ₁ (rad)', fontsize=12)
    axes[0, 0].set_title('Angle θ₁ vs Time', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot theta2
    axes[0, 1].plot(t_true, q_true[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[0, 1].plot(t_pred, q_pred[:, 1], 'r--', linewidth=2, label='PINN Prediction', alpha=0.8)
    axes[0, 1].set_xlabel('Time (s)', fontsize=12)
    axes[0, 1].set_ylabel('θ₂ (rad)', fontsize=12)
    axes[0, 1].set_title('Angle θ₂ vs Time', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot omega1
    axes[1, 0].plot(t_true, qdot_true[:, 0], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[1, 0].plot(t_pred, qdot_pred[:, 0], 'r--', linewidth=2, label='PINN Prediction', alpha=0.8)
    axes[1, 0].set_xlabel('Time (s)', fontsize=12)
    axes[1, 0].set_ylabel('ω₁ (rad/s)', fontsize=12)
    axes[1, 0].set_title('Angular Velocity ω₁ vs Time', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot omega2
    axes[1, 1].plot(t_true, qdot_true[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    axes[1, 1].plot(t_pred, qdot_pred[:, 1], 'r--', linewidth=2, label='PINN Prediction', alpha=0.8)
    axes[1, 1].set_xlabel('Time (s)', fontsize=12)
    axes[1, 1].set_ylabel('ω₂ (rad/s)', fontsize=12)
    axes[1, 1].set_title('Angular Velocity ω₂ vs Time', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()
    
    return plot_path


def compute_metrics(q_true, q_pred, qdot_true=None, qdot_pred=None):
    """Compute error metrics between true and predicted trajectories."""
    metrics = {}
    
    # Position metrics
    mse_pos = np.mean((q_true - q_pred) ** 2)
    mae_pos = np.mean(np.abs(q_true - q_pred))
    rmse_pos = np.sqrt(mse_pos)
    
    metrics['mse_position'] = float(mse_pos)
    metrics['mae_position'] = float(mae_pos)
    metrics['rmse_position'] = float(rmse_pos)
    
    # Per-angle metrics
    for i, angle in enumerate(['theta1', 'theta2']):
        mse_angle = np.mean((q_true[:, i] - q_pred[:, i]) ** 2)
        mae_angle = np.mean(np.abs(q_true[:, i] - q_pred[:, i]))
        metrics[f'mse_{angle}'] = float(mse_angle)
        metrics[f'mae_{angle}'] = float(mae_angle)
    
    # Velocity metrics if available
    if qdot_true is not None and qdot_pred is not None:
        mse_vel = np.mean((qdot_true - qdot_pred) ** 2)
        mae_vel = np.mean(np.abs(qdot_true - qdot_pred))
        metrics['mse_velocity'] = float(mse_vel)
        metrics['mae_velocity'] = float(mae_vel)
    
    return metrics


def run_inference(checkpoint_path, initial_state=None, t_span=(0, 10), num_points=1000,
                 output_dir="data/inference_results", compare_ground_truth=True,
                 parameters=None, device='cpu'):
    """
    Run inference with trained model and optionally compare with ground truth.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        initial_state: [theta1_0, theta2_0, omega1_0, omega2_0]. If None, uses random
        t_span: (t_start, t_end)
        num_points: Number of time points
        output_dir: Directory to save results
        compare_ground_truth: Whether to compute ground truth for comparison
        parameters: Physical parameters dict (m1, m2, l1, l2, g)
        device: Device to run on
    """
    # Default parameters
    if parameters is None:
        parameters = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81}
    
    # Default initial state
    if initial_state is None:
        initial_state = [
            np.random.uniform(-np.pi/4, np.pi/4),  # theta1
            np.random.uniform(-np.pi/4, np.pi/4),  # theta2
            np.random.uniform(-0.5, 0.5),          # omega1
            np.random.uniform(-0.5, 0.5)           # omega2
        ]
    
    print(f"Loading model from: {checkpoint_path}")
    model, cfg, norm_params = load_model(checkpoint_path, device)
    
    print(f"Normalization settings:")
    print(f"  Time: {norm_params['normalize_time']} | Range: [{norm_params['t_min']:.3f}, {norm_params['t_max']:.3f}]")
    print(f"  Angles: {norm_params['normalize_angles']} | Range: [{norm_params['theta_min']:.3f}, {norm_params['theta_max']:.3f}]")
    
    print(f"\nInitial state: theta1={initial_state[0]:.3f}, theta2={initial_state[1]:.3f}, "
          f"omega1={initial_state[2]:.3f}, omega2={initial_state[3]:.3f}")
    print(f"Time span: {t_span}, Points: {num_points}")
    
    # Simulate with PINN using proper normalization
    print("\nSimulating with PINN...")
    t_pred, q_pred, qdot_pred = simulate_with_pinn(
        model, initial_state, t_span, num_points,
        normalize_time=norm_params['normalize_time'],
        normalize_angles=norm_params['normalize_angles'],
        t_min=norm_params['t_min'],
        t_max=norm_params['t_max'],
        theta_min=norm_params['theta_min'],
        theta_max=norm_params['theta_max'],
        device=device
    )
    
    # Save PINN predictions
    save_trajectory(output_dir, t_pred, q_pred, qdot_pred, 
                   prefix="trajectory_pinn", parameters=parameters)
    
    # Compute and compare with ground truth if requested
    if compare_ground_truth:
        print("\nComputing ground truth...")
        t_true, q_true, qdot_true = simulate_ground_truth(
            initial_state, t_span, num_points, **parameters
        )
        
        # Save ground truth
        save_trajectory(output_dir, t_true, q_true, qdot_true,
                       prefix="trajectory_true", parameters=parameters)
        
        # Plot comparison
        print("\nGenerating comparison plots...")
        plot_comparison(t_true, q_true, qdot_true, t_pred, q_pred, qdot_pred, output_dir)
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_metrics(q_true, q_pred, qdot_true, qdot_pred)
        
        # Compute energy drift
        E_true = compute_energy(q_true, qdot_true, **parameters)
        E_pred = compute_energy(q_pred, qdot_pred, **parameters)
        
        metrics['energy_drift_true'] = float(np.abs(E_true - E_true[0]).max() / np.abs(E_true[0]))
        metrics['energy_drift_pred'] = float(np.abs(E_pred - E_pred[0]).max() / np.abs(E_pred[0]))
        
        # Save metrics
        metrics_filename = os.path.join(output_dir, "metrics.json")
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics: {metrics_filename}")
        
        # Print summary
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        print(f"Position RMSE:  {metrics['rmse_position']:.6f}")
        print(f"Position MAE:   {metrics['mae_position']:.6f}")
        if 'mse_velocity' in metrics:
            print(f"Velocity MSE:   {metrics['mse_velocity']:.6f}")
        print(f"\nEnergy drift (true): {metrics['energy_drift_true']*100:.3f}%")
        print(f"Energy drift (pred): {metrics['energy_drift_pred']*100:.3f}%")
        print("="*60)
    
    print(f"\nAll results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained PINN model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='data/inference_results',
                       help='Output directory for results')
    parser.add_argument('--t-start', type=float, default=0.0, help='Start time')
    parser.add_argument('--t-end', type=float, default=10.0, help='End time')
    parser.add_argument('--num-points', type=int, default=1000, help='Number of time points')
    parser.add_argument('--theta1', type=float, default=None, help='Initial theta1')
    parser.add_argument('--theta2', type=float, default=None, help='Initial theta2')
    parser.add_argument('--omega1', type=float, default=None, help='Initial omega1')
    parser.add_argument('--omega2', type=float, default=None, help='Initial omega2')
    parser.add_argument('--no-ground-truth', action='store_true',
                       help='Skip ground truth comparison')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup initial state
    if all(x is not None for x in [args.theta1, args.theta2, args.omega1, args.omega2]):
        initial_state = [args.theta1, args.theta2, args.omega1, args.omega2]
    else:
        initial_state = None  # Will use random
    
    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        initial_state=initial_state,
        t_span=(args.t_start, args.t_end),
        num_points=args.num_points,
        output_dir=args.output_dir,
        compare_ground_truth=not args.no_ground_truth,
        device=args.device
    )


if __name__ == "__main__":
    main()
