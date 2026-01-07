import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import matplotlib.animation as animation


def load_parameters(params_file):
    """Load physical parameters from JSON file."""
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params


def load_trajectory(traj_file):
    """Load trajectory data from NPZ file."""
    data = np.load(traj_file)
    return data['t'], data['q'], data['qdot']


def pendulum_positions(q, l1, l2):
    """Convert angles to Cartesian coordinates for visualization."""
    theta1, theta2 = q[:, 0], q[:, 1]
    
    # First pendulum
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    
    # Second pendulum
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    
    return x1, y1, x2, y2


def plot_trajectories(data_dir='data/raw'):
    """Plot all trajectories over time."""
    data_path = Path(data_dir)
    
    # Load parameters
    params = load_parameters(data_path / 'parameters.json')
    l1, l2 = params['l1'], params['l2']
    
    # Find all trajectory files
    traj_files = sorted(data_path.glob('trajectory_*.npz'))
    
    if not traj_files:
        print("No trajectory files found!")
        return
    
    print(f"Found {len(traj_files)} trajectory files")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Double Pendulum Trajectories vs Time', fontsize=16)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(traj_files)))
    
    for idx, traj_file in enumerate(traj_files):
        t, q, qdot = load_trajectory(traj_file)
        x1, y1, x2, y2 = pendulum_positions(q, l1, l2)
        color = colors[idx]
        label = f'Traj {idx}'
        
        # Plot angles over time
        axes[0, 0].plot(t, q[:, 0], alpha=0.7, color=color, label=label)
        axes[0, 1].plot(t, q[:, 1], alpha=0.7, color=color, label=label)
        
        # Plot angular velocities over time
        axes[1, 0].plot(t, qdot[:, 0], alpha=0.7, color=color, label=label)
        axes[1, 1].plot(t, qdot[:, 1], alpha=0.7, color=color, label=label)
        
        # Plot Cartesian positions over time
        axes[2, 0].plot(t, x2, alpha=0.7, color=color, linestyle='-', label=f'{label}')
        axes[2, 0].plot(t, y2, alpha=0.7, color=color, linestyle='--')
        
        # Plot energy (kinetic + potential approximation)
        KE = 0.5 * params['m1'] * (l1 * qdot[:, 0])**2 + 0.5 * params['m2'] * ((l1 * qdot[:, 0])**2 + (l2 * qdot[:, 1])**2)
        PE = -params['m1'] * params['g'] * y1 - params['m2'] * params['g'] * y2
        E = KE + PE
        axes[2, 1].plot(t, E, alpha=0.7, color=color, label=label)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('θ1 (rad)')
    axes[0, 0].set_title('Angle θ1 vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('θ2 (rad)')
    axes[0, 1].set_title('Angle θ2 vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('θ̇1 (rad/s)')
    axes[1, 0].set_title('Angular Velocity θ̇1 vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('θ̇2 (rad/s)')
    axes[1, 1].set_title('Angular Velocity θ̇2 vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Position (m)')
    axes[2, 0].set_title('End Effector Position vs Time (solid: x, dashed: y)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Energy (J)')
    axes[2, 1].set_title('Total Energy vs Time')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('trajectory_vs_time.png', dpi=150, bbox_inches='tight')
    print("Saved trajectory_vs_time.png")
    plt.show()


def animate_trajectory(traj_idx=0, data_dir='data/raw', save_format='gif', frame_skip=5):
    """Create an animation of a specific trajectory.
    
    Args:
        traj_idx: Index of trajectory to animate
        data_dir: Directory containing trajectory data
        save_format: 'gif' or 'mp4' (mp4 requires ffmpeg)
        frame_skip: Use every Nth frame (higher = faster generation, smaller file)
    """
    data_path = Path(data_dir)
    
    # Load parameters and trajectory
    params = load_parameters(data_path / 'parameters.json')
    l1, l2 = params['l1'], params['l2']
    
    traj_file = data_path / f'trajectory_{traj_idx:03d}.npz'
    if not traj_file.exists():
        print(f"Trajectory file {traj_file} not found!")
        return
    
    t, q, qdot = load_trajectory(traj_file)
    
    # Downsample for performance and memory efficiency
    t = t[::frame_skip]
    q = q[::frame_skip]
    qdot = qdot[::frame_skip]
    
    x1, y1, x2, y2 = pendulum_positions(q, l1, l2)
    
    # Set up the figure (smaller size for memory efficiency)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Animation plot
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Double Pendulum Animation - Trajectory {traj_idx}')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    
    line, = ax1.plot([], [], 'o-', lw=2, color='blue', markersize=8)
    trace, = ax1.plot([], [], '-', lw=1, alpha=0.3, color='red')
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    
    # Phase space plot
    ax2.plot(q[:, 0], qdot[:, 0], alpha=0.3, color='blue', label='θ1')
    ax2.plot(q[:, 1], qdot[:, 1], alpha=0.3, color='red', label='θ2')
    point1, = ax2.plot([], [], 'o', color='blue', markersize=8)
    point2, = ax2.plot([], [], 'o', color='red', markersize=8)
    ax2.set_xlabel('θ (rad)')
    ax2.set_ylabel('θ̇ (rad/s)')
    ax2.set_title('Phase Space')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    trace_x, trace_y = [], []
    
    def init():
        line.set_data([], [])
        trace.set_data([], [])
        point1.set_data([], [])
        point2.set_data([], [])
        time_text.set_text('')
        return line, trace, point1, point2, time_text
    
    def animate(i):
        # Update pendulum arms
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        
        # Update trace
        trace_x.append(x2[i])
        trace_y.append(y2[i])
        if len(trace_x) > 200:  # Keep last 200 points
            trace_x.pop(0)
            trace_y.pop(0)
        trace.set_data(trace_x, trace_y)
        
        # Update phase space points
        point1.set_data([q[i, 0]], [qdot[i, 0]])
        point2.set_data([q[i, 1]], [qdot[i, 1]])
        
        # Update time text
        time_text.set_text(f'Time: {t[i]:.2f} s')
        
        return line, trace, point1, point2, time_text
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=20, blit=True
    )
    
    plt.tight_layout()
    
    # Save animation
    print(f"Creating animation for trajectory {traj_idx}...")
    print(f"Using {len(t)} frames (downsampled from original by factor of {frame_skip})")
    
    if save_format == 'mp4':
        # MP4 is more memory efficient but requires ffmpeg
        try:
            anim.save(
                f'double_pendulum_animation_{traj_idx}.mp4',
                writer='ffmpeg',
                fps=30,
                dpi=80,
                bitrate=1800
            )
            print(f"Saved double_pendulum_animation_{traj_idx}.mp4")
        except Exception as e:
            print(f"Error saving MP4: {e}")
            print("Try installing ffmpeg or use save_format='gif'")
    else:
        # GIF with optimized settings
        anim.save(
            f'double_pendulum_animation_{traj_idx}.gif',
            writer='pillow',
            fps=15,  # Lower FPS for smaller file
            dpi=80   # Lower DPI to reduce memory usage
        )
        print(f"Saved double_pendulum_animation_{traj_idx}.gif")
    
    plt.close(fig)  # Free memory
    print("Animation complete!")


if __name__ == '__main__':
    # Create animation for the first trajectory
    # Use frame_skip to control speed/memory (higher = faster generation)
    # Use save_format='mp4' if you have ffmpeg installed (more efficient)
    animate_trajectory(0, frame_skip=10, save_format='gif')
    
    # Optionally plot all trajectories
    # plot_trajectories()
