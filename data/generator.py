from scipy.integrate import solve_ivp
import numpy as np
import os
import json
from physics.equations import double_pendulum_derivatives, compute_energy

def generate_trajectory(
    initial_state,
    t_span,
    num_points,
    m1=1.0,
    m2=1.0,
    l1=1.0,
    l2=1.0,
    g=9.81
):
    """
    Generate a double pendulum trajectory using numerical integration.
    
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

def generate_dataset(
    output_dir="data/raw",
    num_trajectories=5,
    num_points=1000,
    t_span=(0, 10),
    random_seed=42,
    check_energy=True
):
    """Generate multiple trajectories with random initial conditions."""
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    parameters = {
        'm1': 1.0, 'm2': 1.0,
        'l1': 1.0, 'l2': 1.0,
        'g': 9.81
    }
    
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=2)
    
    print(f"Generating {num_trajectories} trajectories...")
    
    for i in range(num_trajectories):
        initial_state = [
            np.random.uniform(-np.pi/2, np.pi/2),  # theta1: smaller angles
            np.random.uniform(-np.pi/2, np.pi/2),  # theta2: smaller angles
            np.random.uniform(-1, 1),               # omega1: lower velocities
            np.random.uniform(-1, 1)                # omega2: lower velocities
        ]
        
        t, q, qdot = generate_trajectory(initial_state, t_span, num_points, **parameters)
        
        if check_energy:
            E = compute_energy(q, qdot, **parameters)
            E_drift = np.abs(E - E[0]).max()
            E_rel_drift = E_drift / np.abs(E[0])
            print(f"  Trajectory {i+1}: Energy drift = {E_rel_drift*100:.3f}%")
        
        filename = os.path.join(output_dir, f"trajectory_{i:03d}.npz")
        np.savez(filename, t=t, q=q, qdot=qdot)
        print(f"  Saved: {filename}")
    
    print(f"\nDataset complete! Saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    print("=== Generating Dataset ===")
    generate_dataset(num_trajectories=3, num_points=3000, t_span=(0, 10))