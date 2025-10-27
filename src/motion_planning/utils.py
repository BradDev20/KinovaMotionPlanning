import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict
from contextlib import contextmanager


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