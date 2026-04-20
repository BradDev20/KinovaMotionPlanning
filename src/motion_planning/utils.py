import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict
from contextlib import contextmanager
from typing import Tuple


@dataclass
class Obstacle:
    """Dataclass representing a spherical obstacle"""
    center: np.ndarray
    radius: float
    safe_distance: float = 0.0

    def __post_init__(self):
        """Ensure center is a numpy array"""
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center)

    @property
    def danger_threshold(self) -> float:
        """Total threshold distance (radius + safety margin)"""
        return self.radius + self.safe_distance

class PillarObstacle(Obstacle):
    """Dataclass representing a cylindrical pillar obstacle"""

    def __init__(self, center: np.ndarray, radius: float, height: float, safe_distance: float = 0.0):
        super().__init__(center, radius, safe_distance)
        self.height = height

    @property
    def danger_threshold(self) -> float:
        """Total threshold distance for the pillar (radius + safety margin)"""
        return super().danger_threshold


@dataclass
class PerformanceTimer:
    """Performance timer for tracking time spent in different operations"""
    timings: Dict[str, float] = field(default_factory=dict)
    call_counts: Dict[str, int] = field(default_factory=dict)
    _active_timers: Dict[str, float] = field(default_factory=dict)
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time an operation"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            if operation_name not in self.timings:
                self.timings[operation_name] = 0.0
                self.call_counts[operation_name] = 0
            self.timings[operation_name] += elapsed
            self.call_counts[operation_name] += 1
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of timing statistics"""
        total_time = sum(self.timings.values())
        summary = {
            'total_time': total_time,
            'timings': self.timings.copy(),
            'call_counts': self.call_counts.copy(),
            'percentages': {
                name: (time_val / total_time * 100) if total_time > 0 else 0.0
                for name, time_val in self.timings.items()
            },
            'average_times': {
                name: (self.timings[name] / self.call_counts[name]) if self.call_counts[name] > 0 else 0.0
                for name in self.timings.keys()
            }
        }
        return summary
    
    def print_summary(self, title: str = "Performance Summary"):
        """Print formatted timing summary"""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Total Time: {summary['total_time']:.3f}s")
        print(f"{'-'*60}")
        print(f"{'Operation':<25} {'Time (s)':<12} {'%':<8} {'Calls':<10} {'Avg (ms)':<10}")
        print(f"{'-'*60}")
        
        # Sort by time spent (descending)
        sorted_ops = sorted(
            self.timings.keys(),
            key=lambda x: self.timings[x],
            reverse=True
        )
        
        for op in sorted_ops:
            time_val = self.timings[op]
            pct = summary['percentages'][op]
            calls = self.call_counts[op]
            avg_ms = summary['average_times'][op] * 1000
            print(f"{op:<25} {time_val:<12.3f} {pct:<8.1f} {calls:<10} {avg_ms:<10.3f}")
        
        print(f"{'='*60}\n")
    
    def reset(self):
        """Reset all timings"""
        self.timings.clear()
        self.call_counts.clear()
        self._active_timers.clear()

@dataclass    
class Shelf:
    """Backless two-bay shelf with 3 vertical walls + floor + roof."""
    origin: Tuple[float, float, float] = (0.58, -0.10, 0.20)  # front-left-bottom
    depth: float = 0.30
    height: float = 0.20
    bay_widths: Tuple[float, float] = (0.17, 0.40)            # (narrow, wide)
    wall_thickness: float = 0.008
    slab_thickness: float = 0.008
    rgba: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 0.35)
    collidable: bool = True
    body_name: str = "shelf"

    def x_bounds(self): x0, _, _ = self.origin; return (x0, x0 + self.depth)
    def y_bounds(self):
        _, y0, _ = self.origin; wA, wB = self.bay_widths
        return (y0, y0 + wA + wB)
    def z_bounds(self): _, _, z0 = self.origin; return (z0, z0 + self.height)
    def bay_intervals(self):
        _, y0, _ = self.origin; wA, wB = self.bay_widths
        yL, yM, yR = y0, y0 + wA, y0 + wA + wB
        return (yL, yM), (yM, yR)  # (bayA, bayB)
    def bay_center(self, which="A"):
        (x0,x1),(z0,z1) = self.x_bounds(), self.z_bounds()
        bayA, bayB = self.bay_intervals()
        y_min,y_max = bayA if which.upper()=="A" else bayB
        return (0.5*(x0+x1), 0.5*(y_min+y_max), 0.5*(z0+z1))

    def to_xml(self) -> str:
        # geometry helpers
        x0, y0, z0 = self.origin
        wA, wB = self.bay_widths
        total_w = wA + wB
        x_c = x0 + self.depth / 2.0
        z_c = z0 + self.height / 2.0

        # choose the collision bit used by the shelf geoms:
        # keep "1" if you don't care; set to "2" if you plan to collide only with the end-effector
        ct = "1" if self.collidable else "0"

        rgba = f"{self.rgba[0]} {self.rgba[1]} {self.rgba[2]} {self.rgba[3]}"
        hx = self.depth / 2.0
        hy_wall = self.wall_thickness / 2.0
        hz = self.height / 2.0
        hy_slab = total_w / 2.0
        hz_slab = self.slab_thickness / 2.0

        # y locations of the three vertical walls
        yL = y0
        yM = y0 + wA
        yR = y0 + total_w
        yc_mid = y0 + total_w / 2.0

        wall_left = f'''
        <body name="shelf_wall_left_body"  pos="{x_c:.3f} {yL:.3f} {z_c:.3f}">
            <geom name="shelf_wall_left" type="box"
                size="{hx:.3f} {hy_wall:.3f} {hz:.3f}"
                rgba="{rgba}" contype="{ct}" conaffinity="{ct}" margin="0.020"/>
        </body>'''

        wall_mid = f'''
        <body name="shelf_wall_mid_body"   pos="{x_c:.3f} {yM:.3f} {z_c:.3f}">
            <geom name="shelf_wall_mid" type="box"
                size="{hx:.3f} {hy_wall:.3f} {hz:.3f}"
                rgba="{rgba}" contype="{ct}" conaffinity="{ct}" margin="0.020"/>
        </body>'''

        wall_right = f'''
        <body name="shelf_wall_right_body" pos="{x_c:.3f} {yR:.3f} {z_c:.3f}">
            <geom name="shelf_wall_right" type="box"
                size="{hx:.3f} {hy_wall:.3f} {hz:.3f}"
                rgba="{rgba}" contype="{ct}" conaffinity="{ct}" margin="0.020"/>
        </body>'''

        floor = f'''
        <body name="shelf_floor_body" pos="{x_c:.3f} {yc_mid:.3f} {z0 + hz_slab:.3f}">
            <geom name="shelf_floor" type="box"
                size="{hx:.3f} {hy_slab:.3f} {hz_slab:.3f}"
                rgba="{rgba}" contype="{ct}" conaffinity="{ct}" margin="0.020"/>
        </body>'''

        roof = f'''
        <body name="shelf_roof_body"  pos="{x_c:.3f} {yc_mid:.3f} {z0 + self.height - hz_slab:.3f}">
            <geom name="shelf_roof" type="box"
                size="{hx:.3f} {hy_slab:.3f} {hz_slab:.3f}"
                rgba="{rgba}" contype="{ct}" conaffinity="{ct}" margin="0.020"/>
        </body>'''

        return f'''
        <!-- Backless two-bay shelf -->
        <body name="{self.body_name}" pos="0 0 0">
        {wall_left}
        {wall_mid}
        {wall_right}
        {floor}
        {roof}
        </body>'''
    