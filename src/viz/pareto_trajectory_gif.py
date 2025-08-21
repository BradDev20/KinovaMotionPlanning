#!/usr/bin/env python3
"""
Pareto Trajectory GIF Generator

Creates an animated GIF showing the progression of trajectories during Pareto search.
Each frame shows:
1. Left subplot: Trajectory trace in MuJoCo scene (arm static, dots showing end-effector path)
2. Right subplot: Pareto front with current trajectory point highlighted

Usage:
    python pareto_trajectory_gif.py --experiment-dir path/to/experiment --output output.gif
"""

import sys
import os
import numpy as np
import argparse
import json
import pickle
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import mujoco
from PIL import Image
import io

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../examples'))

from motion_planning.kinematics import KinematicsSolver
from motion_planning.utils import Obstacle
from scene_builder import create_standard_scene
from trajectory_loader import TrajectoryLoader


class MuJoCoRenderer:
    """Handles MuJoCo scene rendering and screenshot capture"""
    
    def __init__(self, experiment_config: Dict):
        """Initialize the renderer with experiment configuration"""
        self.config = experiment_config
        self.model = None
        self.data = None
        self.kinematics = None
        self._setup_scene()
        
    def _setup_scene(self):
        """Setup MuJoCo scene based on experiment configuration"""
        # Recreate obstacles from config
        obstacles = []
        for obs_data in self.config['obstacles']:
            obstacles.append(Obstacle(
                center=np.array(obs_data['center']),
                radius=obs_data['radius'],
                safe_distance=obs_data['safe_distance']
            ))
        
        # Create scene
        target_position = np.array(self.config['target_position'])
        scene_path = create_standard_scene(
            obstacles=obstacles,
            target_position=target_position,
            trace_dot_count=1000,  # Large number for trajectory traces
            output_filename="pareto_gif_scene.xml"
        )
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.kinematics = KinematicsSolver(scene_path)
        
        # Set initial position
        start_config = np.array(self.config['start_config'])
        self.data.qpos[:7] = start_config
        self.data.ctrl[:7] = start_config
        mujoco.mj_forward(self.model, self.data)
        
    def render_trajectory_trace(self, trajectory: np.ndarray, color: np.ndarray) -> np.ndarray:
        """
        Render trajectory trace and capture screenshot.
        
        Args:
            trajectory: Trajectory waypoints (N, 7)
            color: RGBA color for trace dots
            
        Returns:
            Screenshot as numpy array
        """
        # Clear existing trace dots
        self._clear_trace_dots()
        
        # Convert trajectory to end-effector positions
        ee_positions = []
        for waypoint in trajectory:
            ee_pos, _ = self.kinematics.forward_kinematics(waypoint)
            ee_positions.append(ee_pos)
        
        # Update trace dots
        for i, pos in enumerate(ee_positions):
            if i >= 1000:  # Limit to available dots
                break
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{i}")
                if body_id >= 0:
                    self.model.body_pos[body_id] = pos
                    # Update color
                    geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"trace_dot_{i}_geom")
                    if geom_id >= 0:
                        self.model.geom_rgba[geom_id] = color
            except:
                pass
        
        # Update simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Render scene
        return self._capture_screenshot()
    
    def _clear_trace_dots(self):
        """Hide all trace dots by moving them below ground"""
        for i in range(1000):
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trace_dot_{i}")
                if body_id >= 0:
                    self.model.body_pos[body_id] = [0, 0, -10]
            except:
                pass
    
    def _capture_screenshot(self) -> np.ndarray:
        """Capture screenshot of current MuJoCo scene"""
        # Create renderer
        renderer = mujoco.Renderer(self.model, height=400, width=400)
        
        # Set camera position for good view
        renderer.update_scene(self.data)
        
        # Render and get image
        pixels = renderer.render()
        
        # Convert to proper format and rotate 180 degrees
        image_array = np.flipud(pixels)  # MuJoCo renders upside down
        image_array = np.rot90(image_array, 2)  # Additional 180 degree rotation
        
        renderer.close()
        return image_array


class ParetoFrontPlotter:
    """Handles Pareto front plotting with highlighted points"""
    
    def __init__(self, trajectory_metadata: List[Dict]):
        """Initialize with trajectory metadata"""
        self.metadata = trajectory_metadata
        self.alphas = np.array([t['alpha'] for t in trajectory_metadata])
        self.length_costs = np.array([t['length_cost'] for t in trajectory_metadata])
        self.obstacle_costs = np.array([t['obstacle_cost'] for t in trajectory_metadata])
        self.colors = np.array([t['color'] for t in trajectory_metadata])
        
        # Setup plotting parameters
        self.cmap = plt.cm.plasma
        self.norm = Normalize(vmin=self.alphas.min(), vmax=self.alphas.max())
        
    def create_pareto_plot(self, highlight_index: int, figsize: Tuple[float, float] = (3.5, 3.0)) -> plt.Figure:
        """
        Create Pareto front plot with highlighted point.
        
        Args:
            highlight_index: Index of trajectory to highlight
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # Plot all points with low opacity
        scatter_colors = self.cmap(self.norm(self.alphas))
        scatter_colors[:, 3] = 0.3  # Low opacity for all points
        
        ax.scatter(self.length_costs, self.obstacle_costs,
                  c=scatter_colors,
                  s=25,
                  marker='D',
                  edgecolor='k',
                  linewidth=0.6,
                  zorder=2)
        
        # Highlight current point
        if 0 <= highlight_index < len(self.metadata):
            highlight_color = list(self.cmap(self.norm(self.alphas[highlight_index])))
            highlight_color[3] = 1.0  # Full opacity
            
            ax.scatter(self.length_costs[highlight_index], 
                      self.obstacle_costs[highlight_index],
                      c=[highlight_color],
                      s=100,
                      marker='D',
                      edgecolor='red',
                      linewidth=2.0,
                      zorder=5)
            
            # Add alpha label
            alpha_val = self.alphas[highlight_index]
            ax.annotate(f'α = {alpha_val:.1f}',
                       xy=(self.length_costs[highlight_index], self.obstacle_costs[highlight_index]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       zorder=6)
        
        # Colorbar removed - alpha label provides sufficient information
        
        # Styling
        self._style_axes(ax)
        
        # Title
        ax.text(0.95, 1.1, "Trade-offs between objectives",
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=8,
                family='serif',
                color='black')
        
        plt.tight_layout()
        return fig
    
    def _style_axes(self, ax):
        """Apply consistent styling to axes"""
        # Calculate axis limits with extra padding on the right for alpha labels
        x_margin = (self.length_costs.max() - self.length_costs.min()) * 0.1
        y_margin = (self.obstacle_costs.max() - self.obstacle_costs.min()) * 0.1
        
        x_min = self.length_costs.min() - x_margin
        x_max = self.length_costs.max() + x_margin * 3  # Extra padding on right for alpha labels
        y_min = self.obstacle_costs.min() - y_margin  
        y_max = self.obstacle_costs.max() + y_margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Style spines
        axis_color = '0.4'
        axis_linewidth = 2.0
        
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(axis_linewidth)
            ax.spines[spine].set_color(axis_color)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        # Add appropriate ticks and labels
        # Length axis (x) - create 4-5 evenly spaced ticks
        x_range = x_max - x_min
        x_tick_values = np.linspace(self.length_costs.min(), self.length_costs.max(), 4)
        ax.set_xticks(x_tick_values)
        ax.set_xticklabels([f'{val:.1f}' for val in x_tick_values], fontsize=8, family='serif')
        
        # Closeness axis (y) - create 4-5 evenly spaced ticks  
        y_range = y_max - y_min
        y_tick_values = np.linspace(self.obstacle_costs.min(), self.obstacle_costs.max(), 4)
        ax.set_yticks(y_tick_values)
        ax.set_yticklabels([f'{val:.1f}' for val in y_tick_values], fontsize=8, family='serif')
        
        ax.set_xlabel("Length", labelpad=7, fontsize=10, family='serif')
        ax.set_ylabel("Closeness", labelpad=7, fontsize=10, family='serif')
        
        # Add arrow heads
        arrow_style = dict(arrowstyle='-|>', color=axis_color, linewidth=axis_linewidth)
        arrow_size = 10
        ax.add_patch(FancyArrowPatch((x_max, y_min), (x_max + x_margin*0.3, y_min),
                                    **arrow_style, mutation_scale=arrow_size))
        ax.add_patch(FancyArrowPatch((x_min, y_max), (x_min, y_max + y_margin*0.3),
                                    **arrow_style, mutation_scale=arrow_size))


class ParetoTrajectoryGifGenerator:
    """Main class for generating Pareto trajectory GIFs"""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize the GIF generator.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = experiment_dir
        self.loader = TrajectoryLoader(experiment_dir)
        self.renderer = MuJoCoRenderer(self.loader.config)
        self.plotter = ParetoFrontPlotter(self.loader.metadata)
        
    def generate_gif(self, output_path: str, duration: float = 1.0, 
                    show_progress: bool = True) -> None:
        """
        Generate the animated GIF.
        
        Args:
            output_path: Path to save the GIF
            duration: Duration per frame in seconds
            show_progress: Whether to show progress
        """
        if show_progress:
            print(f"Generating GIF with {len(self.loader.available_trajectories)} frames...")
        
        frames = []
        
        for i, traj_info in enumerate(self.loader.available_trajectories):
            if show_progress:
                print(f"Processing trajectory {i+1}/{len(self.loader.available_trajectories)}: {traj_info['trajectory_id']}")
            
            # Load trajectory
            trajectory_data = self.loader.load_trajectory(traj_info['trajectory_id'])
            if trajectory_data is None:
                continue
                
            # Create frame
            frame = self._create_frame(trajectory_data, i)
            frames.append(frame)
        
        if frames:
            # Save as GIF
            if show_progress:
                print(f"Saving GIF to {output_path}...")
            
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(duration * 1000),  # PIL uses milliseconds
                loop=0
            )
            
            if show_progress:
                print(f"GIF saved successfully! ({len(frames)} frames)")
        else:
            print("No frames generated!")
    
    def _create_frame(self, trajectory_data: Dict, highlight_index: int) -> Image.Image:
        """
        Create a single frame combining MuJoCo render and Pareto plot.
        
        Args:
            trajectory_data: Trajectory data dictionary
            highlight_index: Index for highlighting in Pareto plot
            
        Returns:
            PIL Image of the combined frame
        """
        # Render MuJoCo trajectory trace
        trajectory = trajectory_data['trajectory']
        color = np.array(trajectory_data['color'])
        mujoco_image = self.renderer.render_trajectory_trace(trajectory, color)
        
        # Create Pareto plot
        pareto_fig = self.plotter.create_pareto_plot(highlight_index)
        
        # Convert matplotlib figure to image
        pareto_buffer = io.BytesIO()
        pareto_fig.savefig(pareto_buffer, format='png', dpi=150, bbox_inches='tight')
        pareto_buffer.seek(0)
        pareto_image = Image.open(pareto_buffer)
        plt.close(pareto_fig)
        
        # Convert MuJoCo image to PIL
        mujoco_pil = Image.fromarray(mujoco_image.astype(np.uint8))
        
        # Resize images to same height
        target_height = 400
        mujoco_pil = mujoco_pil.resize((target_height, target_height))
        
        # Calculate pareto image size to maintain aspect ratio
        pareto_aspect = pareto_image.width / pareto_image.height
        pareto_width = int(target_height * pareto_aspect)
        pareto_image = pareto_image.resize((pareto_width, target_height))
        
        # Combine images side by side
        total_width = target_height + pareto_width
        combined = Image.new('RGB', (total_width, target_height), 'white')
        combined.paste(mujoco_pil, (0, 0))
        combined.paste(pareto_image, (target_height, 0))
        
        return combined


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Pareto trajectory GIF visualization')
    
    parser.add_argument('--experiment-dir', type=str, required=True,
                        help='Path to experiment directory containing saved trajectories')
    parser.add_argument('--output', type=str, default='pareto_trajectories.gif',
                        help='Output GIF filename (default: pareto_trajectories.gif)')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Duration per frame in seconds (default: 1.0)')
    parser.add_argument('--output-dir', type=str, default='src/viz',
                        help='Output directory (default: src/viz)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Check experiment directory
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        return
    
    # Setup output path
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)
    
    try:
        # Create generator and run
        print(f"Loading experiment from: {args.experiment_dir}")
        generator = ParetoTrajectoryGifGenerator(args.experiment_dir)
        
        print(f"Found {len(generator.loader.available_trajectories)} trajectories to process")
        
        generator.generate_gif(
            output_path=output_path,
            duration=args.duration,
            show_progress=True
        )
        
    except Exception as e:
        print(f"Error generating GIF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 