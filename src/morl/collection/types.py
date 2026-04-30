"""
Common types and helpers for tracking dataset collection.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class CollectionTaskDispatch:
    """
    Tells a worker which task to work on and how many times to try.
    """
    task: object
    task_index: int
    alpha_values: tuple[float, ...]
    restart_count: int
    mode: str
    order_offset: int

@dataclass(frozen=True)
class PlannerCoordinatorStop:
    """
    Special signal to kill off a worker process.
    """
    worker_id: int

@dataclass
class CollectionJobResult:
    """
    What we get back after a single planning run.
    Either we got a record (success!) or a failure payload.
    """
    order_index: int
    task_index: int
    record: dict[str, object] | None = None
    failure: dict[str, object] | None = None

class CollectionProgressTracker:
    """
    Fancy progress bar that shows how we're doing on both the current task 
    and the overall collection run. 
    """
    def __init__(
        self,
        *,
        tasks: list,
        alpha_values: list[float],
        restart_count: int,
    ) -> None:
        self.task_ids = [str(task.task_id) for task in tasks]
        self.task_count = len(self.task_ids)
        self.jobs_per_task = len(alpha_values) * int(restart_count)
        self.total_jobs = self.task_count * self.jobs_per_task
        self.completed_total = 0
        self.success_total = 0
        self.failure_total = 0
        self.completed_by_task = [0 for _ in range(self.task_count)]
        self.last_task_index = 0
        self._rendered = False
        self._last_line_length = 0
        # Use stdout directly so we can overwrite lines with \r
        self._stream = sys.__stdout__ if getattr(sys, "__stdout__", None) is not None else sys.stdout

    def start(self) -> None:
        """Kicks off the rendering."""
        self._render()

    def advance(self, result: CollectionJobResult) -> None:
        """Bump the numbers when a job finishes."""
        if self.total_jobs <= 0:
            return
        self.completed_total += 1
        if result.failure is None:
            self.success_total += 1
        else:
            self.failure_total += 1
        task_index = int(result.task_index)
        if 0 <= task_index < self.task_count:
            self.completed_by_task[task_index] += 1
            self.last_task_index = task_index
        self._render()

    def finish(self) -> None:
        """Cleanup after we're all done."""
        if self._rendered:
            print(file=self._stream, flush=True)
            self._rendered = False
            self._last_line_length = 0

    def _render(self) -> None:
        """The math for the bars and percentages."""
        task_index = min(max(int(self.last_task_index), 0), max(self.task_count - 1, 0))
        task_done = self.completed_by_task[task_index] if self.completed_by_task else 0
        task_total = max(self.jobs_per_task, 1)
        total_done = self.completed_total
        total_jobs = max(self.total_jobs, 1)
        task_bar = self._bar(task_done, task_total)
        total_bar = self._bar(total_done, total_jobs)
        task_percent = 100.0 * float(task_done) / float(task_total)
        total_percent = 100.0 * float(total_done) / float(total_jobs)
        task_label = self.task_ids[task_index] if self.task_ids else "task"
        line = (
            f"Task {task_label} {task_bar} {task_done}/{task_total} ({task_percent:5.1f}%) | "
            f"Total {total_bar} {total_done}/{total_jobs} ({total_percent:5.1f}%) | "
            f"ok={self.success_total} fail={self.failure_total}"
        )
        padded_line = line.ljust(self._last_line_length)
        print(f"\r{padded_line}", end="", file=self._stream, flush=True)
        self._rendered = True
        self._last_line_length = max(self._last_line_length, len(line))

    @staticmethod
    def _bar(done: int, total: int, width: int = 18) -> str:
        """Draw a progress bar with hashtags."""
        total = max(int(total), 1)
        done = max(0, min(int(done), total))
        filled = int(round(width * (float(done) / float(total))))
        return "[" + "#" * filled + "-" * (width - filled) + "]"
