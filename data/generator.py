from scipy.integrate import solve_ivp
import numpy as np
import os
import json
from physics.equations import double_pendulum_derivatives

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
        args=(m1, m2, l1, l2, g),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    
    t = solution.t
    q = solution.y[:2, :].T  # [theta1, theta2]
    qdot = solution.y[2:, :].T  # [omega1, omega2]
    
    return t, q, qdot


def compute_energy(q, qdot, m1, m2, l1, l2, g):
    """
    Compute total energy for verification.
    """
    theta1, theta2 = q[:, 0], q[:, 1]
    omega1, omega2 = qdot[:, 0], qdot[:, 1]
    
    # Center of mass positions
    y1 = -l1/2 * np.cos(theta1)
    y2 = -l1 * np.cos(theta1) - l2/2 * np.cos(theta2)
    
    # Potential energy
    U = m1 * g * y1 + m2 * g * y2
    
    # Kinetic energy
    vx1 = l1/2 * omega1 * np.cos(theta1)
    vy1 = l1/2 * omega1 * np.sin(theta1)
    vx2 = l1 * omega1 * np.cos(theta1) + l2/2 * omega2 * np.cos(theta2)
    vy2 = l1 * omega1 * np.sin(theta1) + l2/2 * omega2 * np.sin(theta2)
    
    K_trans = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
    K_rot = (1/12) * m1 * l1**2 * omega1**2 + (1/12) * m2 * l2**2 * omega2**2
    
    return K_trans + K_rot + U



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
            np.random.uniform(-np.pi, np.pi),  # theta1
            np.random.uniform(-np.pi, np.pi),  # theta2
            np.random.uniform(-2, 2),          # omega1
            np.random.uniform(-2, 2)           # omega2
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
    generate_dataset(num_trajectories=3, num_points=1000)