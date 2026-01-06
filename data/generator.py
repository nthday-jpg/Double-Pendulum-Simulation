import numpy as np
import json
import os
from scipy.integrate import solve_ivp

def double_pendulum_derivatives(t, y, m1, m2, l1, l2, g):
    """
    Compute derivatives for double pendulum system.
    
    State vector y = [theta1, theta2, omega1, omega2]
    Returns: dy/dt = [omega1, omega2, alpha1, alpha2]
    """
    theta1, theta2, omega1, omega2 = y
    
    delta = theta2 - theta1
    
    # Mass matrix terms
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
    den2 = (l2 / l1) * den1
    
    # Angular accelerations
    alpha1 = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) +
              m2 * g * np.sin(theta2) * np.cos(delta) +
              m2 * l2 * omega2**2 * np.sin(delta) -
              (m1 + m2) * g * np.sin(theta1)) / den1
    
    alpha2 = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) +
              (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
              (m1 + m2) * l1 * omega1**2 * np.sin(delta) -
              (m1 + m2) * g * np.sin(theta2)) / den2
    
    return [omega1, omega2, alpha1, alpha2]

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
    Generate a double pendulum trajectory.
    
    Args:
        initial_state: [theta1_0, theta2_0, omega1_0, omega2_0]
        t_span: (t_start, t_end)
        num_points: Number of time points
        m1, m2: Masses
        l1, l2: Lengths
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
    
    t = solution.t
    q = solution.y[:2, :].T  # [theta1, theta2]
    qdot = solution.y[2:, :].T  # [omega1, omega2]
    
    return t, q, qdot

def generate_dataset(
    output_dir="data/raw",
    num_trajectories=5,
    num_points=1000,
    t_span=(0, 10),
    random_seed=42
):
    """
    Generate multiple trajectories with random initial conditions.
    
    Args:
        output_dir: Directory to save data
        num_trajectories: Number of trajectories to generate
        num_points: Points per trajectory
        t_span: Time span for each trajectory
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed parameters
    parameters = {
        'm1': 1.0,
        'm2': 1.0,
        'l1': 1.0,
        'l2': 1.0,
        'g': 9.81
    }
    
    # Save parameters
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=2)
    
    print(f"Generating {num_trajectories} trajectories...")
    
    for i in range(num_trajectories):
        # Random initial conditions
        # Angles: -π to π, Angular velocities: -2 to 2
        theta1_0 = np.random.uniform(-np.pi, np.pi)
        theta2_0 = np.random.uniform(-np.pi, np.pi)
        omega1_0 = np.random.uniform(-2, 2)
        omega2_0 = np.random.uniform(-2, 2)
        
        initial_state = [theta1_0, theta2_0, omega1_0, omega2_0]
        
        # Generate trajectory
        t, q, qdot = generate_trajectory(
            initial_state,
            t_span,
            num_points,
            **parameters
        )
        
        # Save trajectory
        filename = os.path.join(output_dir, f"trajectory_{i:03d}.npz")
        np.savez(filename, t=t, q=q, qdot=qdot)
        
        print(f"  Saved trajectory {i+1}/{num_trajectories}: {filename}")
        print(f"    Initial: θ1={theta1_0:.3f}, θ2={theta2_0:.3f}, "
              f"ω1={omega1_0:.3f}, ω2={omega2_0:.3f}")
    
    print(f"\nDataset generation complete! Saved to {output_dir}")
    return output_dir

def generate_simple_test_data(
    output_path="data/raw/test_trajectory.npz",
    parameters_path="data/raw/parameters.json"
):
    """
    Generate a single simple trajectory for quick testing.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simple initial condition
    initial_state = [np.pi/4, np.pi/6, 0.0, 0.0]  # Small angles, at rest
    
    parameters = {
        'm1': 1.0,
        'm2': 1.0,
        'l1': 1.0,
        'l2': 1.0,
        'g': 9.81
    }
    
    t, q, qdot = generate_trajectory(
        initial_state,
        t_span=(0, 5),
        num_points=500,
        **parameters
    )
    
    # Save
    np.savez(output_path, t=t, q=q, qdot=qdot)
    
    with open(parameters_path, "w") as f:
        json.dump(parameters, f, indent=2)
    
    print(f"Test data generated:")
    print(f"  Trajectory: {output_path}")
    print(f"  Parameters: {parameters_path}")
    print(f"  Time points: {len(t)}")
    print(f"  Time range: [{t[0]:.2f}, {t[-1]:.2f}]")
    
    return output_path, parameters_path

if __name__ == "__main__":
    # Generate test dataset
    print("=== Generating Simple Test Data ===")
    generate_simple_test_data()
    
    print("\n=== Generating Full Dataset ===")
    generate_dataset(num_trajectories=1, num_points=1000)