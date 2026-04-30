"""
Parallel and sequential execution logic for collecting trajectories.
This is where the actual multiprocessing happens, including the 
worker processes and the batch coordinator.
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty

import numpy as np

from ...motion_planning.torch_trajopt import TorchPlannerJob, TorchPlannerResult
from ..planning import (
    PlannerConfig,
    build_torch_planner_job,
    finalize_planned_trajectory,
    prepare_task_planning_context,
    run_torch_planner_batch,
)
from .seed_bank import (
    MAX_SEEDS_PER_FAMILY,
    SEED_NOISE_SCALE,
    SeedEntry,
    _adapt_seed_trajectory,
    _promote_task_seeds,
    _select_diverse_risky_seed,
    _select_extreme_seed,
    _select_family_seed,
)
from .types import (
    CollectionJobResult,
    CollectionProgressTracker,
    CollectionTaskDispatch,
    PlannerCoordinatorStop,
)

TASK_PROBE_RESTART_COUNT = 4

def _chunked(items: list, size: int):
    """Helper to split a list into smaller chunks for batching."""
    chunk_size = max(int(size), 1)
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]

def _build_task_jobs(
    dispatch: CollectionTaskDispatch,
    *,
    context,
    planner_config: PlannerConfig,
    worker_id: int,
) -> list[tuple[float, int, TorchPlannerJob]]:
    """Constructs the full list of planner jobs for a specific task and alpha values."""
    jobs: list[tuple[float, int, TorchPlannerJob]] = []
    for restart_index in range(dispatch.restart_count):
        for alpha_index, alpha in enumerate(dispatch.alpha_values):
            order_index = dispatch.order_offset + alpha_index * dispatch.restart_count + restart_index
            jobs.append(
                (
                    float(alpha),
                    int(restart_index),
                    build_torch_planner_job(
                        context,
                        alpha=float(alpha),
                        mode=dispatch.mode,
                        restart_index=restart_index,
                        order_index=order_index,
                        worker_id=worker_id,
                        planner_config=planner_config,
                    ),
                )
            )
    return jobs

def _probe_job_count(dispatch: CollectionTaskDispatch) -> int:
    """Number of 'probe' jobs to try before deciding if a task is a lost cause."""
    return len(dispatch.alpha_values) * min(int(dispatch.restart_count), TASK_PROBE_RESTART_COUNT)

def _task_has_success(results: list[CollectionJobResult]) -> bool:
    """Did at least one job for this task actually work?"""
    return any(result.record is not None for result in results)

def _skip_remaining_task_jobs(
    dispatch: CollectionTaskDispatch,
    *,
    job_specs: list[tuple[float, int, TorchPlannerJob]],
    attempted_jobs: int,
    failure_factory,
) -> list[CollectionJobResult]:
    """Marks the rest of the jobs for a task as skipped if the probes all failed."""
    total_jobs = len(job_specs)
    skipped_specs = job_specs[attempted_jobs:]
    return [
        CollectionJobResult(
            order_index=job.order_index,
            task_index=dispatch.task_index,
            failure=failure_factory(
                dispatch.task,
                alpha=alpha,
                restart_index=restart_index,
                attempted_jobs=attempted_jobs,
                total_jobs=total_jobs,
            ),
        )
        for alpha, restart_index, job in skipped_specs
    ]

def _finalize_job_results(
    *,
    dispatch: CollectionTaskDispatch,
    context,
    job_specs: list[tuple[float, int, TorchPlannerJob]],
    responses: dict[str, TorchPlannerResult],
    planner_config: PlannerConfig,
    failure_factory,
) -> list[CollectionJobResult]:
    """Takes the raw planner results and turns them into saved trajectory records."""
    results: list[CollectionJobResult] = []
    for alpha, restart_index, job in job_specs:
        planner_result = responses[job.request_id]
        try:
            record = finalize_planned_trajectory(
                context,
                planner_result=planner_result,
                alpha=alpha,
                mode=dispatch.mode,
                restart_index=restart_index,
                planner_config=planner_config,
            )
            results.append(CollectionJobResult(order_index=job.order_index, task_index=dispatch.task_index, record=record))
        except Exception as exc:
            results.append(
                CollectionJobResult(
                    order_index=job.order_index,
                    task_index=dispatch.task_index,
                    failure=failure_factory(context.task, alpha, restart_index, exc),
                )
            )
    return results

def _collect_task_sequential(
    dispatch: CollectionTaskDispatch,
    *,
    scene_dir: Path,
    planner_config: PlannerConfig,
    failure_factory,
    skipped_failure_factory,
    mute_stdio_context,
    seed_bank_by_family: dict[str, list[SeedEntry]] | None = None,
) -> list[CollectionJobResult]:
    """Runs planning for a single task one step at a time (no multiprocessing)."""
    try:
        context = prepare_task_planning_context(dispatch.task, scene_dir=scene_dir, planner_config=planner_config)
    except Exception as exc:
        return [
            CollectionJobResult(
                order_index=dispatch.order_offset + alpha_index * dispatch.restart_count + restart_index,
                task_index=dispatch.task_index,
                failure=failure_factory(dispatch.task, alpha, restart_index, exc),
            )
            for alpha_index, alpha in enumerate(dispatch.alpha_values)
            for restart_index in range(dispatch.restart_count)
        ]

    job_specs = _build_task_jobs(dispatch, context=context, planner_config=planner_config, worker_id=-1)
    probe_count = _probe_job_count(dispatch)
    probe_job_specs = job_specs[:probe_count]
    remaining_job_specs = job_specs[probe_count:]

    # Apply family seeds to warm-start our first few tries
    if seed_bank_by_family is not None:
        family = str(dispatch.task.family)
        seeds = seed_bank_by_family.get(family, [])
        if seeds:
            rng = np.random.default_rng(int(dispatch.task.planner_seed))
            start_cfg = np.asarray(dispatch.task.start_config, dtype=np.float32)
            goal_cfg = np.asarray(context.goal_config, dtype=np.float32)
            safe_seed = _select_extreme_seed(seeds, kind="safe")
            risky_seed = _select_diverse_risky_seed(seeds, safe_seed=safe_seed)

            updated_probe: list[tuple[float, int, TorchPlannerJob]] = []
            for alpha, restart_index, job in probe_job_specs:
                if int(restart_index) == 0:
                    updated_probe.append((alpha, restart_index, job))
                    continue
                style = "random"
                seed = None
                if int(restart_index) == 1:
                    style = "safe"
                    seed = safe_seed
                elif int(restart_index) == 2:
                    style = "risky"
                    seed = risky_seed
                if seed is None:
                    seed = _select_family_seed(seeds, start_config=start_cfg, goal_config=goal_cfg, rng=rng)
                if seed is None:
                    updated_probe.append((alpha, restart_index, job))
                    continue
                try:
                    warm = _adapt_seed_trajectory(seed, start_config=start_cfg, goal_config=goal_cfg)
                    updated_probe.append(
                        (
                            alpha,
                            restart_index,
                            dataclasses.replace(
                                job,
                                warm_start_trajectory=warm,
                                warm_start_noise_scale=float(SEED_NOISE_SCALE),
                                warm_start_tag=f"family_seed_{style}",
                            ),
                        )
                    )
                except Exception:
                    updated_probe.append((alpha, restart_index, job))
            probe_job_specs = updated_probe

    responses: dict[str, TorchPlannerResult] = {}
    for batch in _chunked(probe_job_specs, planner_config.gpu_batch_size):
        batch_results = run_torch_planner_batch([job for _, _, job in batch], planner_config=planner_config)
        responses.update({result.request_id: result for result in batch_results})
    
    finalized_probe = _finalize_job_results(
        dispatch=dispatch,
        context=context,
        job_specs=probe_job_specs,
        responses=responses,
        planner_config=planner_config,
        failure_factory=failure_factory,
    )

    task_seeds: tuple[SeedEntry, SeedEntry] | None = None
    if seed_bank_by_family is not None and _task_has_success(finalized_probe):
        successful_records = [item.record for item in finalized_probe if item.record is not None]
        task_seeds = _promote_task_seeds(
            seed_bank_by_family=seed_bank_by_family,
            family=str(dispatch.task.family),
            successful_records=successful_records,
            default_task_id=str(dispatch.task.task_id),
        )

    if remaining_job_specs and not _task_has_success(finalized_probe):
        return finalized_probe + _skip_remaining_task_jobs(
            dispatch,
            job_specs=job_specs,
            attempted_jobs=probe_count,
            failure_factory=skipped_failure_factory,
        )

    if seed_bank_by_family is not None and remaining_job_specs and task_seeds is not None:
        safe_seed_entry, risky_seed_entry = task_seeds
        start_cfg = np.asarray(dispatch.task.start_config, dtype=np.float32)
        goal_cfg = np.asarray(context.goal_config, dtype=np.float32)
        updated_remaining: list[tuple[float, int, TorchPlannerJob]] = []
        for alpha, restart_index, job in remaining_job_specs:
            chosen = safe_seed_entry if float(alpha) <= 0.5 else risky_seed_entry
            style = "safe" if chosen is safe_seed_entry else "risky"
            try:
                warm = _adapt_seed_trajectory(chosen, start_config=start_cfg, goal_config=goal_cfg)
                updated_remaining.append(
                    (
                        alpha,
                        restart_index,
                        dataclasses.replace(
                            job,
                            warm_start_trajectory=warm,
                            warm_start_noise_scale=float(SEED_NOISE_SCALE),
                            warm_start_tag=f"task_seed_{style}",
                        ),
                    )
                )
            except Exception:
                updated_remaining.append((alpha, restart_index, job))
        remaining_job_specs = updated_remaining

    for batch in _chunked(remaining_job_specs, planner_config.gpu_batch_size):
        batch_results = run_torch_planner_batch([job for _, _, job in batch], planner_config=planner_config)
        responses.update({result.request_id: result for result in batch_results})
    
    return finalized_probe + _finalize_job_results(
        dispatch=dispatch,
        context=context,
        job_specs=remaining_job_specs,
        responses=responses,
        planner_config=planner_config,
        failure_factory=failure_factory,
    )

def _planner_coordinator_main(
    request_queue,
    response_queues,
    worker_count: int,
    planner_config: PlannerConfig,
    mute_stdio_context,
    quiet: bool = False,
) -> None:
    """
    The 'brains' of the GPU planning. 
    It sits and waits for jobs from workers, bundles them into batches, 
    runs them on the GPU, and sends the results back.
    """
    with mute_stdio_context(bool(quiet)):
        stop_count = 0
        while stop_count < worker_count:
            try:
                first_item = request_queue.get(timeout=0.1)
            except Empty:
                continue
            if isinstance(first_item, PlannerCoordinatorStop):
                stop_count += 1
                continue

            batch: list[TorchPlannerJob] = [first_item]
            deadline = time.perf_counter() + (float(planner_config.gpu_batch_timeout_ms) / 1000.0)
            while len(batch) < int(planner_config.gpu_batch_size):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    queued_item = request_queue.get(timeout=remaining)
                except Empty:
                    break
                if isinstance(queued_item, PlannerCoordinatorStop):
                    stop_count += 1
                    continue
                batch.append(queued_item)

            for result in run_torch_planner_batch(batch, planner_config=planner_config):
                response_queues[result.worker_id].put(result)

def _collection_worker_main(
    worker_id: int,
    task_queue,
    planner_request_queue,
    planner_response_queue,
    result_queue,
    scene_dir: str,
    planner_config: PlannerConfig,
    failure_factory,
    skipped_failure_factory,
    mute_stdio_context,
    quiet: bool = False,
    initial_seed_bank: dict[str, list[SeedEntry]] | None = None,
) -> None:
    """
    A worker process that grinds through tasks. 
    It requests planning jobs from the coordinator and reports results 
    back to the main process.
    """
    with mute_stdio_context(bool(quiet)):
        seed_bank_by_family: dict[str, list[SeedEntry]] = {
            family: list(seeds) for family, seeds in (initial_seed_bank or {}).items()
        }
        while True:
            dispatch = task_queue.get()
            if dispatch is None:
                planner_request_queue.put(PlannerCoordinatorStop(worker_id=worker_id))
                return

            family = str(dispatch.task.family)
            seeds = seed_bank_by_family.get(family, [])
            
            # If we already have plenty of seeds, don't waste time on long repairs
            task_planner_config = planner_config
            if len(seeds) >= MAX_SEEDS_PER_FAMILY:
                task_planner_config = dataclasses.replace(
                    planner_config,
                    repair_max_iter=15,
                    repair_max_fun=50,
                )

            try:
                context = prepare_task_planning_context(dispatch.task, scene_dir=scene_dir, planner_config=task_planner_config)
            except Exception as exc:
                for alpha_index, alpha in enumerate(dispatch.alpha_values):
                    for restart_index in range(dispatch.restart_count):
                        result_queue.put(
                            CollectionJobResult(
                                order_index=dispatch.order_offset + alpha_index * dispatch.restart_count + restart_index,
                                task_index=dispatch.task_index,
                                failure=failure_factory(dispatch.task, alpha, restart_index, exc),
                            )
                        )
                continue

            job_specs = _build_task_jobs(dispatch, context=context, planner_config=planner_config, worker_id=worker_id)
            probe_count = _probe_job_count(dispatch)
            probe_job_specs = job_specs[:probe_count]
            remaining_job_specs = job_specs[probe_count:]

            # Try to warm-start with family seeds
            family = str(dispatch.task.family)
            seeds = seed_bank_by_family.get(family, [])
            if seeds:
                rng = np.random.default_rng(int(dispatch.task.planner_seed))
                start_cfg = np.asarray(dispatch.task.start_config, dtype=np.float32)
                goal_cfg = np.asarray(context.goal_config, dtype=np.float32)
                safe_seed = _select_extreme_seed(seeds, kind="safe")
                risky_seed = _select_diverse_risky_seed(seeds, safe_seed=safe_seed)
                updated_probe: list[tuple[float, int, TorchPlannerJob]] = []
                for alpha, restart_index, job in probe_job_specs:
                    if int(restart_index) == 0:
                        updated_probe.append((alpha, restart_index, job))
                        continue
                    style = "random"
                    seed = None
                    if int(restart_index) == 1:
                        style = "safe"
                        seed = safe_seed
                    elif int(restart_index) == 2:
                        style = "risky"
                        seed = risky_seed
                    if seed is None:
                        seed = _select_family_seed(seeds, start_config=start_cfg, goal_config=goal_cfg, rng=rng)
                    if seed is None:
                        updated_probe.append((alpha, restart_index, job))
                        continue
                    try:
                        warm = _adapt_seed_trajectory(seed, start_config=start_cfg, goal_config=goal_cfg)
                        updated_probe.append(
                            (
                                alpha,
                                restart_index,
                                dataclasses.replace(
                                    job,
                                    warm_start_trajectory=warm,
                                    warm_start_noise_scale=float(SEED_NOISE_SCALE),
                                    warm_start_tag=f"family_seed_{style}",
                                ),
                            )
                        )
                    except Exception:
                        updated_probe.append((alpha, restart_index, job))
                probe_job_specs = updated_probe
            
            for _, _, job in probe_job_specs:
                planner_request_queue.put(job)

            responses: dict[str, TorchPlannerResult] = {}
            while len(responses) < len(probe_job_specs):
                planner_result = planner_response_queue.get()
                responses[planner_result.request_id] = planner_result

            finalized_probe = _finalize_job_results(
                dispatch=dispatch,
                context=context,
                job_specs=probe_job_specs,
                responses=responses,
                planner_config=planner_config,
                failure_factory=failure_factory,
            )
            for finalized in finalized_probe:
                result_queue.put(finalized)

            task_seeds: tuple[SeedEntry, SeedEntry] | None = None
            if _task_has_success(finalized_probe):
                successful_records = [item.record for item in finalized_probe if item.record is not None]
                task_seeds = _promote_task_seeds(
                    seed_bank_by_family=seed_bank_by_family,
                    family=family,
                    successful_records=successful_records,
                    default_task_id=str(dispatch.task.task_id),
                )
            if remaining_job_specs and not _task_has_success(finalized_probe):
                for skipped in _skip_remaining_task_jobs(
                    dispatch,
                    job_specs=job_specs,
                    attempted_jobs=probe_count,
                    failure_factory=skipped_failure_factory,
                ):
                    result_queue.put(skipped)
                continue

            if remaining_job_specs and task_seeds is not None:
                safe_seed_entry, risky_seed_entry = task_seeds
                start_cfg = np.asarray(dispatch.task.start_config, dtype=np.float32)
                goal_cfg = np.asarray(context.goal_config, dtype=np.float32)
                updated_remaining: list[tuple[float, int, TorchPlannerJob]] = []
                for alpha, restart_index, job in remaining_job_specs:
                    chosen = safe_seed_entry if float(alpha) <= 0.5 else risky_seed_entry
                    style = "safe" if chosen is safe_seed_entry else "risky"
                    try:
                        warm = _adapt_seed_trajectory(chosen, start_config=start_cfg, goal_config=goal_cfg)
                        updated_remaining.append(
                            (
                                alpha,
                                restart_index,
                                dataclasses.replace(
                                    job,
                                    warm_start_trajectory=warm,
                                    warm_start_noise_scale=float(SEED_NOISE_SCALE),
                                    warm_start_tag=f"task_seed_{style}",
                                ),
                            )
                        )
                    except Exception:
                        updated_remaining.append((alpha, restart_index, job))
                remaining_job_specs = updated_remaining

            for _, _, job in remaining_job_specs:
                planner_request_queue.put(job)
            while len(responses) < len(job_specs):
                planner_result = planner_response_queue.get()
                responses[planner_result.request_id] = planner_result

            for finalized in _finalize_job_results(
                dispatch=dispatch,
                context=context,
                job_specs=remaining_job_specs,
                responses=responses,
                planner_config=planner_config,
                failure_factory=failure_factory,
            ):
                result_queue.put(finalized)

def _collect_task_results_parallel(
    tasks: list,
    *,
    alpha_values: list[float],
    restart_count: int,
    scene_dir: Path,
    planner_mode: str,
    planner_config: PlannerConfig,
    num_workers: int,
    failure_factory,
    skipped_failure_factory,
    mute_stdio_context,
    progress_tracker: CollectionProgressTracker | None = None,
    initial_seed_bank: dict[str, list[SeedEntry]] | None = None,
    quiet: bool = False,
) -> list[CollectionJobResult]:
    """Sets up the multiprocessing pool and runs all the tasks in parallel."""
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    planner_request_queue = ctx.Queue()
    result_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(num_workers)]
    jobs_per_task = len(alpha_values) * restart_count
    coordinator = None
    workers: list[mp.Process] = []

    def _close_queue(queue) -> None:
        try:
            queue.close()
        except Exception:
            pass
        try:
            queue.cancel_join_thread()
        except Exception:
            pass

    def _shutdown_processes() -> None:
        active_processes = [process for process in [coordinator, *workers] if process is not None]
        for process in active_processes:
            if process.is_alive():
                process.terminate()
        for process in active_processes:
            try:
                process.join(timeout=5.0)
            except Exception:
                pass
        for process in active_processes:
            if process.is_alive() and hasattr(process, "kill"):
                try:
                    process.kill()
                except Exception:
                    pass
        for process in active_processes:
            try:
                process.join(timeout=1.0)
            except Exception:
                pass

    try:
        coordinator = ctx.Process(
            target=_planner_coordinator_main,
            args=(planner_request_queue, response_queues, num_workers, planner_config, mute_stdio_context, bool(quiet)),
        )
        workers = [
            ctx.Process(
                target=_collection_worker_main,
                args=(
                    worker_id,
                    task_queue,
                    planner_request_queue,
                    response_queues[worker_id],
                    result_queue,
                    str(scene_dir),
                    planner_config,
                    failure_factory,
                    skipped_failure_factory,
                    mute_stdio_context,
                    bool(quiet),
                    initial_seed_bank,
                ),
            )
            for worker_id in range(num_workers)
        ]
        coordinator.start()
        for worker in workers:
            worker.start()

        for task_index, task in enumerate(tasks):
            task_queue.put(
                CollectionTaskDispatch(
                    task=task,
                    task_index=task_index,
                    alpha_values=tuple(alpha_values),
                    restart_count=int(restart_count),
                    mode=planner_mode,
                    order_offset=task_index * jobs_per_task,
                )
            )
        for _ in workers:
            task_queue.put(None)

        total_job_count = len(tasks) * jobs_per_task
        collected_results = []
        for _ in range(total_job_count):
            result = result_queue.get()
            collected_results.append(result)
            if progress_tracker is not None:
                progress_tracker.advance(result)

        for worker in workers:
            worker.join()
        coordinator.join()
        return collected_results
    except BaseException:
        _shutdown_processes()
        raise
    finally:
        _close_queue(task_queue)
        _close_queue(planner_request_queue)
        _close_queue(result_queue)
        for queue in response_queues:
            _close_queue(queue)
