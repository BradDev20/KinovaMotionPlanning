#!/usr/bin/env python3
"""
Simple demonstration of switching between linear weighted sum and weighted maximum cost formulations.

This script shows the professional, modular design for cost function composition.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from motion_planning.cost_functions import (
    TrajectoryLengthCostFunction,
    ObstacleAvoidanceCostFunction, 
    CompositeCostFunction,
    CostModeFactory
)
from motion_planning.utils import Obstacle


def demonstrate_cost_modes():
    """Demonstrate switching between sum and max cost formulations"""
    
    print("🧪 Composite Cost Function Demonstration")
    print("=" * 50)
    
    # Create some dummy cost functions for demonstration
    length_cost = TrajectoryLengthCostFunction(weight=1.0)
    
    obstacles = [
        Obstacle(center=np.array([0.5, 0.0, 0.5]), radius=0.1, safe_distance=0.05)
    ]
    
    # Note: This would normally require a kinematics solver
    # For demo purposes, we'll just show the interface
    print("Cost Functions Available:")
    print("  1. TrajectoryLengthCostFunction")
    print("  2. ObstacleAvoidanceCostFunction")
    print()
    
    # Demonstration weights
    weights = [10.0, 100.0]  # [length_weight, obstacle_weight]
    
    print("🔧 Creating Composite Cost Functions:")
    print()
    
    # Method 1: Direct instantiation
    print("Method 1: Direct Instantiation")
    print("-" * 30)
    
    # Linear weighted sum
    sum_cost = CompositeCostFunction(
        cost_functions=[length_cost],  # Would include obstacle cost with real kinematics
        weights=[10.0],
        mode='sum'
    )
    print(f"Sum mode: {sum_cost.get_mode_info()}")
    print()
    
    # Weighted maximum with tie-breaking
    max_cost = CompositeCostFunction(
        cost_functions=[length_cost],
        weights=[10.0], 
        mode='max',
        rho=0.01
    )
    print(f"Max mode: {max_cost.get_mode_info()}")
    print()
    
    # Method 2: Factory pattern
    print("Method 2: Factory Pattern")
    print("-" * 30)
    
    safe_cost = CostModeFactory.create_pareto_comparison(
        cost_functions=[length_cost],
        weights=[10.0],
        strategy='safe'  # Uses sum mode
    )
    print(f"Safe strategy: {safe_cost.get_mode_info()}")
    print()
    
    risky_cost = CostModeFactory.create_pareto_comparison(
        cost_functions=[length_cost],
        weights=[10.0],
        strategy='risky'  # Uses max mode
    )
    print(f"Risky strategy: {risky_cost.get_mode_info()}")
    print()
    
    # Method 3: Research mode
    print("Method 3: Research Mode")
    print("-" * 30)
    
    research_cost = CostModeFactory.create_research_mode(
        cost_functions=[length_cost],
        weights=[10.0],
        rho=0.005  # Custom tie-breaking parameter
    )
    print(f"Research mode: {research_cost.get_mode_info()}")
    print()
    
    # Demonstrate mode switching
    print("🔄 Dynamic Mode Switching:")
    print("-" * 30)
    
    dynamic_cost = CompositeCostFunction(
        cost_functions=[length_cost],
        weights=[10.0],
        mode='sum'
    )
    
    print(f"Initial: {dynamic_cost.mode.upper()} mode")
    
    dynamic_cost.switch_mode('max', rho=0.02)
    print(f"After switch: {dynamic_cost.mode.upper()} mode")
    print()
    
    print("✅ All cost formulations configured successfully!")
    print()
    print("Integration with TrajOpt Planner:")
    print("  1. Create planner with cost_mode='composite'")
    print("  2. Use planner.setup_composite_cost() for convenience")
    print("  3. Or use planner.set_composite_cost_function() for custom setup")


if __name__ == "__main__":
    demonstrate_cost_modes() 