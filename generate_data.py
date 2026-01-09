#!/usr/bin/env python3
"""
Generate multiple double pendulum trajectories for training.
Compatible with Kaggle environments.
"""

import argparse
from data.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description='Generate double pendulum dataset')
    
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory for trajectory files')
    parser.add_argument('--num_trajectories', type=int, default=10,
                        help='Number of trajectories to generate')
    parser.add_argument('--num_points', type=int, default=2000,
                        help='Number of time points per trajectory')
    parser.add_argument('--t_start', type=float, default=0.0,
                        help='Start time')
    parser.add_argument('--t_end', type=float, default=5.0,
                        help='End time')
    parser.add_argument('--vary_parameters', action='store_true',
                        help='Vary physics parameters for each trajectory')
    parser.add_argument('--check_energy', action='store_true', default=True,
                        help='Check energy conservation')
    
    # Physics parameters (used if not varying)
    parser.add_argument('--m1', type=float, default=1.0,
                        help='Mass of first pendulum')
    parser.add_argument('--m2', type=float, default=1.0,
                        help='Mass of second pendulum')
    parser.add_argument('--l1', type=float, default=1.0,
                        help='Length of first pendulum')
    parser.add_argument('--l2', type=float, default=1.0,
                        help='Length of second pendulum')
    parser.add_argument('--g', type=float, default=9.81,
                        help='Gravitational acceleration')
    
    args = parser.parse_args()
    
    # Setup parameters
    if args.vary_parameters:
        parameters = None  # Will be varied automatically
    else:
        parameters = {
            'm1': args.m1,
            'm2': args.m2,
            'l1': args.l1,
            'l2': args.l2,
            'g': args.g
        }
    
    # Generate dataset
    generate_dataset(
        output_dir=args.output_dir,
        num_trajectories=args.num_trajectories,
        parameters=parameters,
        vary_parameters=args.vary_parameters,
        num_points=args.num_points,
        t_span=(args.t_start, args.t_end),
        check_energy=args.check_energy
    )


if __name__ == '__main__':
    main()
