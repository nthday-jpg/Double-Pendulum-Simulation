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
    parameters=None,
    vary_parameters=False,
    num_points=1000,
    t_span=(0, 10),
    random_seed=42,
    check_energy=True
):
    """Generate multiple trajectories with random initial conditions.
    
    Args:
        output_dir: Output directory for trajectories
        num_trajectories: Number of trajectories to generate
        parameters: Dict or list of dicts with physical parameters (m1, m2, l1, l2, g).
                   If None, uses default values.
                   If dict, same parameters for all trajectories.
                   If list, must have length num_trajectories (one dict per trajectory).
        vary_parameters: If True and parameters is None/dict, randomly vary parameters
                        for each trajectory within reasonable ranges
        num_points: Number of time points per trajectory
        t_span: Time span (t_start, t_end)
        random_seed: Random seed for reproducibility
        check_energy: Whether to check energy conservation
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Default parameters
    default_params = {
        'm1': 1.0, 'm2': 1.0,
        'l1': 1.0, 'l2': 1.0,
        'g': 9.81
    }
    
    # Setup parameters for each trajectory
    if parameters is None:
        if vary_parameters:
            # Generate random parameters for each trajectory
            params_list = []
            for i in range(num_trajectories):
                params_list.append({
                    'm1': np.random.uniform(0.5, 2.0),
                    'm2': np.random.uniform(0.5, 2.0),
                    'l1': np.random.uniform(0.5, 1.5),
                    'l2': np.random.uniform(0.5, 1.5),
                    'g': 9.81  # Keep gravity constant
                })
        else:
            params_list = [default_params.copy() for _ in range(num_trajectories)]
    elif isinstance(parameters, dict):
        # Single parameter set for all trajectories
        params_list = [parameters.copy() for _ in range(num_trajectories)]
    elif isinstance(parameters, list):
        if len(parameters) != num_trajectories:
            raise ValueError(f"Length of parameters list ({len(parameters)}) must match num_trajectories ({num_trajectories})")
        params_list = parameters
    else:
        raise TypeError("parameters must be None, dict, or list of dicts")
    
    print(f"Generating {num_trajectories} trajectories...")
    if vary_parameters or isinstance(parameters, list):
        print("Using different parameters for each trajectory")
    
    for i in range(num_trajectories):
        traj_params = params_list[i]
        
        # Save parameters for this trajectory
        params_filename = os.path.join(output_dir, f"parameters_{i:03d}.json")
        with open(params_filename, "w") as f:
            json.dump(traj_params, f, indent=2)
        
        initial_state = [
            np.random.uniform(-np.pi/2, np.pi/2),  # theta1: smaller angles
            np.random.uniform(-np.pi/2, np.pi/2),  # theta2: smaller angles
            np.random.uniform(-1, 1),               # omega1: lower velocities
            np.random.uniform(-1, 1)                # omega2: lower velocities
        ]
        
        t, q, qdot = generate_trajectory(initial_state, t_span, num_points, **traj_params)
        
        if check_energy:
            E = compute_energy(q, qdot, **traj_params)
            E_drift = np.abs(E - E[0]).max()
            E_rel_drift = E_drift / np.abs(E[0])
            param_str = f"m1={traj_params['m1']:.2f}, m2={traj_params['m2']:.2f}, l1={traj_params['l1']:.2f}, l2={traj_params['l2']:.2f}"
            print(f"  Trajectory {i:03d} ({param_str}): Energy drift = {E_rel_drift*100:.3f}%")
        
        filename = os.path.join(output_dir, f"trajectory_{i:03d}.npz")
        np.savez(filename, t=t, q=q, qdot=qdot)
        print(f"  Saved: {filename} and {params_filename}")
    
    print(f"\nDataset complete! Saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    
    # Example 1: Generate with same parameters for all trajectories
    generate_dataset(num_trajectories=1, num_points=3000, t_span=(5, 15))
    
    # Example 2: Generate with varied parameters for each trajectory
    # generate_dataset(num_trajectories=3, num_points=3000, t_span=(0, 10), vary_parameters=True)
    
    # Example 3: Generate with custom parameters list
    # custom_params = [
    #     {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81},
    #     {'m1': 1.5, 'm2': 0.8, 'l1': 1.2, 'l2': 0.9, 'g': 9.81},
    #     {'m1': 0.7, 'm2': 1.3, 'l1': 0.8, 'l2': 1.1, 'g': 9.81},
    # ]
    # generate_dataset(num_trajectories=3, parameters=custom_params
    # generate_dataset(num_trajectories=3, num_points=3000, t_span=(0, 10))