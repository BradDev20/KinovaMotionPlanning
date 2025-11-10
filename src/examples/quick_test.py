#!/usr/bin/env python3
"""
Quick test script for debugging - minimal Pareto front with customizable number of alpha values
"""
import sys
import os
import argparse
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from examples.pareto_search import ParetoSearchDemo, SearchConfiguration

def parse_args():
    parser = argparse.ArgumentParser(description='Quick test with uniformly spaced alpha values')
    parser.add_argument('--num-alphas', type=int, default=3,
                       help='Number of uniformly spaced alpha values (default: 3)')
    parser.add_argument('--alpha-start', type=float, default=0.0,
                       help='Starting alpha value (default: 0.0)')
    parser.add_argument('--alpha-end', type=float, default=1.0,
                       help='Ending alpha value (default: 1.0)')
    parser.add_argument('--cost-mode', choices=['sum', 'max', 'max_constrained'], default='sum',
                       help='Cost function formulation (default: sum)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Generate uniformly spaced alpha values in [alpha_start, alpha_end]
    if args.num_alphas == 1:
        # Single point: use the middle of the range
        alpha_values = [(args.alpha_start + args.alpha_end) / 2.0]
        alpha_start = alpha_values[0]
        alpha_end = alpha_values[0]
        alpha_step = 1.0
    elif args.num_alphas == 2:
        # Two points: use start and end exactly
        alpha_values = [args.alpha_start, args.alpha_end]
        alpha_start = args.alpha_start
        alpha_end = args.alpha_end
        alpha_step = args.alpha_end - args.alpha_start
    else:
        # Multiple points: uniformly distribute across the range
        alpha_values = np.linspace(args.alpha_start, args.alpha_end, args.num_alphas)
        alpha_start = args.alpha_start
        alpha_step = (args.alpha_end - args.alpha_start) / (args.num_alphas - 1)
        # Set end to second-to-last value so arange(start, end+step, step) includes alpha_end exactly
        alpha_end = args.alpha_end - alpha_step
    
    config = SearchConfiguration(
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        alpha_step=alpha_step,
        cost_mode=args.cost_mode,
        rho=0.01,
        save_trajectories=True,
        experiment_name='quick_test'
    )
    
    print("=" * 60)
    print(f"QUICK TEST MODE - {args.num_alphas} trajectories")
    print(f"Alpha range: [{args.alpha_start:.2f}, {args.alpha_end:.2f}] with step {alpha_step:.4f}")
    print(f"Alpha values: {[f'{a:.3f}' for a in alpha_values]}")
    print(f"Cost mode: {args.cost_mode.upper()}")
    print("Results will be saved to:")
    print("  src/pareto_data_and_results/quick_test/")
    print("=" * 60)
    
    demo = ParetoSearchDemo(config)
    demo.run_demo()
    
    print("\n" + "=" * 60)
    print("Quick test complete!")
    print(f"Generated {args.num_alphas} trajectories")
    print("Check results in:")
    print("  src/pareto_data_and_results/quick_test/")
    print("  - trajectory_metadata.json")
    print("  - trajectory_alpha_*.pkl")
    print("  - experiment_config.json")
    print("=" * 60)

