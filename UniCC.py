from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math
import time

import numpy as np

from utils import fun_record


Objective = Callable[[np.ndarray], Any]
ObjectiveFactory = Callable[[], Objective]
OptimizerClass = Callable[[Dict[str, Any], Dict[str, Any]], Any]


@dataclass
class UniCCCycleResult:
    """单次 UniCC 运行结果 / Result of one independent UniCC cycle."""

    fitness_record: List[float]
    time_cost: float
    best_individual: np.ndarray


@dataclass
class UniCCResult:
    """多次独立运行的聚合结果 / Aggregated results of multiple independent cycles."""

    cycle_results: List[UniCCCycleResult]

    @property
    def fitness_records(self) -> List[List[float]]:
        return [result.fitness_record for result in self.cycle_results]

    @property
    def time_records(self) -> List[float]:
        return [result.time_cost for result in self.cycle_results]

    @property
    def best_individuals(self) -> List[np.ndarray]:
        return [result.best_individual for result in self.cycle_results]


class UniCC:
    """
    统一协同进化框架 / Unified Cooperative Coevolution framework.

    本类只负责 UniCC 架构本身，不绑定具体 benchmark、分组文件、绘图或保存逻辑。
    This class contains only the UniCC architecture, not any benchmark, grouping
    file, plotting, or persistence logic.

    核心职责 / Core responsibilities:
    - 构造调度矩阵 A / build the scheduling matrix A;
    - 按 window 执行 batch / execute batches by window size W;
    - 按 pipeline layer 传递信息 / transmit information across pipeline layers L;
    - 用 fun_record 包装子空间目标函数 / wrap subspace objectives with fun_record;
    - 支持多次独立运行 / support independent repeated runs.

    实验脚本应在类外提供目标函数、优化器、分组、边界、随机种子和结果保存逻辑。
    Experiments should provide objective, optimizer, decomposition, bounds, seeds,
    and result persistence outside this class.
    """

    def __init__(
        self,
        objective: Optional[Objective],
        optimizer_cls: OptimizerClass,
        grouping_result: Sequence[Sequence[int]],
        lower_boundary: Any,
        upper_boundary: Any,
        max_fes: Optional[float] = None,
        sub_fes: Optional[int] = None,
        interval: int = 0,
        pipeline_layers: int = 1,
        window_size: int = 1,
        optimizer_options: Optional[Dict[str, Any]] = None,
        objective_factory: Optional[ObjectiveFactory] = None,
    ) -> None:
        """
        初始化 UniCC 框架配置 / Initialize UniCC framework settings.

        Args:
            objective: 完整空间目标函数。应接收 2-D batch，并返回每行一个 fitness。
                Full-space objective. It should accept a 2-D batch and return one
                fitness value per row.
            optimizer_cls: 优化器类，需满足 `optimizer_cls(problem, options).optimize()`。
                Optimizer class following `optimizer_cls(problem, options).optimize()`.
            grouping_result: 子空间维度索引列表 / List of subspace dimension indices.
            lower_boundary: 标量、完整边界向量或子空间边界向量下界。
                Scalar, full-vector, or subspace-vector lower bounds.
            upper_boundary: 标量、完整边界向量或子空间边界向量上界。
                Scalar, full-vector, or subspace-vector upper bounds.
            max_fes: 总预算；仅在未显式传入 `sub_fes` 时用于计算子空间预算。
                Total budget, used only to compute `sub_fes` if absent.
            sub_fes: 每个子空间优化任务的函数评价次数。
                Function evaluations allocated to each subspace task.
            interval: UniCC 间隔参数 I / UniCC interval I.
            pipeline_layers: 流水线层数 L / Number of pipeline layers L.
            window_size: 窗口大小 W，即每层最多并行的子空间任务数。
                Window size W, i.e. maximum parallel subspace tasks per layer.
            optimizer_options: 每次子空间优化都会复制使用的基础优化器参数。
                Base optimizer options copied into every subspace optimizer call.
            objective_factory: 当 objective 不能被多进程序列化时，用 factory 在 worker 内创建目标函数。
                Pickle-safe factory for creating a fresh objective inside workers.
        """
        if objective is None and objective_factory is None:
            raise ValueError("Either objective or objective_factory must be provided.")
        if max_fes is None and sub_fes is None:
            raise ValueError("Either max_fes or sub_fes must be provided.")

        _validate_grouping(grouping_result)
        _validate_nonnegative_int("interval", interval)
        _validate_positive_int("pipeline_layers", pipeline_layers)
        _validate_positive_int("window_size", window_size)

        self.objective = objective
        self.objective_factory = objective_factory
        self.optimizer_cls = optimizer_cls
        self.grouping_result = [np.asarray(group, dtype=int) for group in grouping_result]
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.max_fes = max_fes
        self.interval = int(interval)
        self.pipeline_layers = int(pipeline_layers)
        self.window_size = int(window_size)
        self.optimizer_options = dict(optimizer_options or {})
        self.subspace_num = len(self.grouping_result)
        self.sub_fes = int(sub_fes) if sub_fes is not None else int(max_fes // (self.subspace_num * self.pipeline_layers))

        if self.sub_fes <= 0:
            raise ValueError("sub_fes must be positive. Increase max_fes or pass sub_fes explicitly.")

    @property
    def schedule_matrix(self) -> np.ndarray:
        return self.build_schedule_matrix(self.subspace_num, self.interval, self.pipeline_layers)

    @property
    def lag_total(self) -> int:
        return self.calculate_lag_total(
            subspace_num=self.subspace_num,
            pipeline_layers=self.pipeline_layers,
            interval=self.interval,
            window_size=self.window_size,
        )

    @property
    def mean_lagging_subspaces(self) -> float:
        return self.lag_total / float(self.subspace_num * self.pipeline_layers)

    @staticmethod
    def build_schedule_matrix(subspace_num: int, interval: int, pipeline_layers: int) -> np.ndarray:
        """
        构造论文 Algorithm 1 中的调度矩阵 A。
        Build the scheduling matrix A in Algorithm 1.

        A[l, r] = r - I*l, if 0 <= r - I*l < M; otherwise -1.

        其中 M 是子空间数量，I 是 interval，L 是 pipeline_layers。
        Here M is the number of subspaces, I is interval, and L is pipeline_layers.
        """
        _validate_positive_int("subspace_num", subspace_num)
        _validate_nonnegative_int("interval", interval)
        _validate_positive_int("pipeline_layers", pipeline_layers)

        rows = subspace_num + interval * (pipeline_layers - 1)
        matrix = -np.ones((rows, pipeline_layers), dtype=int)

        for layer in range(pipeline_layers):
            offset = interval * layer
            matrix[offset : offset + subspace_num, layer] = np.arange(subspace_num)

        return matrix.T

    @staticmethod
    def calculate_lag_total(subspace_num: int, pipeline_layers: int, interval: int, window_size: int) -> int:
        """
        计算 UniCC 调度对应的总滞后子空间数。
        Calculate the total number of lagging subspaces for the UniCC schedule.

        这里用版本号近似跟踪每个子空间的信息新鲜度。
        A version counter is used to track the information freshness of each subspace.
        """
        _validate_positive_int("subspace_num", subspace_num)
        _validate_positive_int("pipeline_layers", pipeline_layers)
        _validate_nonnegative_int("interval", interval)
        _validate_positive_int("window_size", window_size)

        if subspace_num <= 1:
            return 0

        schedule = UniCC.build_schedule_matrix(subspace_num, interval, pipeline_layers)
        column_num = schedule.shape[1]
        batch_num = math.ceil(column_num / window_size)
        versions = [0] * subspace_num  # 子空间版本号 / Version counter of each subspace.
        lag_total = 0

        for batch_idx in range(batch_num):
            start_col = batch_idx * window_size
            end_col = min((batch_idx + 1) * window_size, column_num)

            for layer in range(pipeline_layers):
                current_tasks = schedule[layer, start_col:end_col]
                current_tasks = current_tasks[current_tasks != -1]

                for subspace_idx in current_tasks:
                    subspace_idx = int(subspace_idx)
                    current_version = versions[subspace_idx]
                    lag_total += sum(
                        versions[other_idx] <= current_version
                        for other_idx in range(subspace_num)
                        if other_idx != subspace_idx
                    )

                for subspace_idx in current_tasks:
                    versions[int(subspace_idx)] += 1

        return lag_total

    def optimize_one_cycle(self, initial_best_individual: np.ndarray, seed: Optional[int] = None) -> UniCCCycleResult:
        """执行一次完整 UniCC 优化 / Run one complete UniCC optimization cycle."""
        return _run_cycle(
            config=self._worker_config(),
            initial_best_individual=np.asarray(initial_best_individual, dtype=float).copy(),
            seed=seed,
        )

    def run(
        self,
        initial_best_individual: np.ndarray,
        cycle_num: int = 1,
        seed: Optional[int] = None,
        parallel_cycles: bool = True,
        max_cycle_workers: Optional[int] = None,
    ) -> UniCCResult:
        """
        执行多次相互独立的 UniCC 运行。
        Run multiple independent UniCC cycles.

        `cycle_num` 沿用实验习惯：每次运行从同一个初始解出发，但使用不同 seed。
        `cycle_num` follows the experimental convention: each cycle starts from the
        same initial individual and uses a different seed.
        """
        _validate_positive_int("cycle_num", cycle_num)
        initial_best_individual = np.asarray(initial_best_individual, dtype=float)
        cycle_seeds = [None if seed is None else int(seed) + idx for idx in range(cycle_num)]
        config = self._worker_config()

        if parallel_cycles and cycle_num > 1:
            worker_count = max_cycle_workers or cycle_num
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(_run_cycle, config, initial_best_individual.copy(), cycle_seeds[idx])
                    for idx in range(cycle_num)
                ]
                cycle_results = [future.result() for future in futures]
        else:
            cycle_results = [
                _run_cycle(config, initial_best_individual.copy(), cycle_seeds[idx])
                for idx in range(cycle_num)
            ]

        return UniCCResult(cycle_results=cycle_results)

    def _worker_config(self) -> Dict[str, Any]:
        """打包供多进程 worker 使用的配置 / Pack process-worker configuration."""
        return {
            "objective": self.objective,
            "objective_factory": self.objective_factory,
            "optimizer_cls": self.optimizer_cls,
            "grouping_result": self.grouping_result,
            "lower_boundary": self.lower_boundary,
            "upper_boundary": self.upper_boundary,
            "sub_fes": self.sub_fes,
            "interval": self.interval,
            "pipeline_layers": self.pipeline_layers,
            "window_size": self.window_size,
            "optimizer_options": self.optimizer_options,
        }


def _run_cycle(config: Dict[str, Any], initial_best_individual: np.ndarray, seed: Optional[int]) -> UniCCCycleResult:
    """执行单次 UniCC 主循环 / Execute the main loop for one UniCC cycle."""
    grouping_result = config["grouping_result"]
    subspace_num = len(grouping_result)
    pipeline_layers = config["pipeline_layers"]
    window_size = config["window_size"]
    schedule = UniCC.build_schedule_matrix(subspace_num, config["interval"], pipeline_layers)

    column_num = schedule.shape[1]
    batch_num = math.ceil(column_num / window_size)
    best_individual = initial_best_individual.copy()
    # 每个子空间当前最新片段，用于 batch 结束后同步回全局 best。
    # Latest segment for each subspace; synchronized back to the global best after each batch.
    sub_best_records = [best_individual[grouping_result[idx]].copy() for idx in range(subspace_num)]
    fitness_record: List[float] = []
    start_time = time.time()

    for batch_idx in range(batch_num):
        start_col = batch_idx * window_size
        end_col = min((batch_idx + 1) * window_size, column_num)
        batch_columns = list(range(start_col, end_col))

        # Col_best in the paper: each active column owns a local background vector
        # inside the current batch. Updates are visible to later layers in this batch.
        # 论文中的 Col_best：当前 batch 内每列维护一个局部背景向量，供后续层读取。
        column_best_map = {
            "best_individual_list": [best_individual.copy() for _ in batch_columns],
            "subspace_index": [-1 for _ in batch_columns],
            "column_index": batch_columns,
        }

        for layer in range(pipeline_layers):
            # 同一层中的 active tasks 并行执行；层与层之间顺序执行。
            # Active tasks in the same layer run in parallel; layers run sequentially.
            active_tasks = _active_tasks(schedule[layer, start_col:end_col], start_col)
            if not active_tasks:
                continue

            with ProcessPoolExecutor(max_workers=min(window_size, len(active_tasks))) as executor:
                futures = []
                for task_idx, (subspace_idx, column_idx) in enumerate(active_tasks):
                    task_seed = _derive_task_seed(seed, batch_idx, layer, task_idx)
                    futures.append(
                        executor.submit(
                            _optimize_subspace,
                            config,
                            column_best_map,
                            subspace_idx,
                            column_idx,
                            task_seed,
                        )
                    )
                layer_results = [future.result() for future in futures]

            for sub_fitness, sub_individual, subspace_idx, column_idx in layer_results:
                # 将子空间优化结果写回它所属的 column best，并记录该子空间最新片段。
                # Write the subspace result back to its column best and cache the latest segment.
                update_idx = column_best_map["column_index"].index(column_idx)
                column_best_map["best_individual_list"][update_idx] = sub_individual
                column_best_map["subspace_index"][update_idx] = subspace_idx
                sub_best_records[subspace_idx] = sub_individual[grouping_result[subspace_idx]].copy()
                fitness_record.extend(sub_fitness)

        # batch 结束后才同步到全局 best，保持论文中 inter-batch update 的语义。
        # Synchronize to the global best only after a batch, matching inter-batch updates.
        for subspace_idx in range(subspace_num):
            best_individual[grouping_result[subspace_idx]] = sub_best_records[subspace_idx]

    objective = _build_objective(config)
    fitness_record.append(_evaluate_full_objective(objective, best_individual))

    return UniCCCycleResult(
        fitness_record=fitness_record,
        time_cost=time.time() - start_time,
        best_individual=best_individual,
    )


def _optimize_subspace(
    config: Dict[str, Any],
    column_best_map: Dict[str, Any],
    subspace_index: int,
    column_index: int,
    seed: Optional[int],
) -> Tuple[List[float], np.ndarray, int, int]:
    """优化单个子空间任务 / Optimize one scheduled subspace task."""
    objective = _build_objective(config)
    grouping_result = config["grouping_result"]
    dims = grouping_result[subspace_index]

    map_idx = column_best_map["column_index"].index(column_index)
    best_individual = column_best_map["best_individual_list"][map_idx].copy()

    # 若同一 batch 中已有该子空间的新结果，则把该子空间片段同步到当前背景向量。
    # If this subspace has been updated in the same batch, reuse its latest segment.
    if subspace_index in column_best_map["subspace_index"]:
        previous_idx = column_best_map["subspace_index"].index(subspace_index)
        previous_best = column_best_map["best_individual_list"][previous_idx]
        best_individual[dims] = previous_best[dims]

    # fun_record 将子空间候选解拼回完整向量后再调用 full objective。
    # fun_record combines subspace candidates with the full background before evaluation.
    wrapped_objective = fun_record(objective, best_individual, dims, CC=True)
    problem = {
        "fitness_function": wrapped_objective,
        "ndim_problem": len(dims),
        "lower_boundary": _slice_boundary(config["lower_boundary"], dims),
        "upper_boundary": _slice_boundary(config["upper_boundary"], dims),
    }
    # 每个任务复制一份 optimizer options，避免 worker 间共享可变字典。
    # Copy optimizer options per task to avoid sharing mutable dictionaries between workers.
    options = dict(config["optimizer_options"])
    options.setdefault("max_function_evaluations", config["sub_fes"])
    options.setdefault("mean", best_individual[dims].copy())
    if seed is not None:
        options.setdefault("seed_rng", int(seed))

    optimizer = config["optimizer_cls"](problem, options)
    result = optimizer.optimize()

    updated_individual = best_individual.copy()
    updated_individual[dims] = _extract_best_x(result)
    fitness_record = [float(value) for value in wrapped_objective.fitness_record]
    return fitness_record, updated_individual, subspace_index, column_index


def _active_tasks(row_slice: np.ndarray, start_col: int) -> List[Tuple[int, int]]:
    """提取当前层当前 batch 的有效任务 / Extract valid tasks in the current layer batch."""
    return [
        (int(subspace_idx), start_col + local_col)
        for local_col, subspace_idx in enumerate(row_slice)
        if int(subspace_idx) != -1
    ]


def _derive_task_seed(seed: Optional[int], batch_idx: int, layer: int, task_idx: int) -> Optional[int]:
    """派生子任务 seed，保证同一 cycle 内任务随机性可复现 / Derive reproducible task seeds."""
    if seed is None:
        return None
    return int(seed) + batch_idx * 100000 + layer * 1000 + task_idx


def _build_objective(config: Dict[str, Any]) -> Objective:
    """获取 full objective；必要时在 worker 内用 factory 创建 / Get or create the full objective."""
    if config["objective_factory"] is not None:
        return config["objective_factory"]()
    return config["objective"]


def _evaluate_full_objective(objective: Objective, individual: np.ndarray) -> float:
    """评价完整解 / Evaluate a full-space individual."""
    try:
        value = objective(individual.reshape(1, -1))
    except Exception:
        value = objective(individual)
    return float(_as_fitness_array(value)[0])


def _as_fitness_array(value: Any) -> np.ndarray:
    """统一 fitness 返回格式为一维数组 / Normalize fitness output to a 1-D array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _slice_boundary(boundary: Any, dims: Sequence[int]) -> np.ndarray:
    """按子空间维度切出边界 / Slice boundary values for a subspace."""
    dims = np.asarray(dims, dtype=int)
    boundary_arr = np.asarray(boundary, dtype=float)
    if boundary_arr.ndim == 0:
        return np.full(len(dims), float(boundary_arr))
    if boundary_arr.size == len(dims):
        return boundary_arr.reshape(-1).copy()
    return boundary_arr.reshape(-1)[dims].copy()


def _extract_best_x(result: Any) -> np.ndarray:
    """从优化器返回值中提取最优子空间解 / Extract best subspace solution from optimizer result."""
    if isinstance(result, dict):
        for key in ("best_so_far_x", "best_x", "x"):
            if key in result:
                return np.asarray(result[key], dtype=float)
    if hasattr(result, "best_so_far_x"):
        return np.asarray(result.best_so_far_x, dtype=float)
    if hasattr(result, "best_x"):
        return np.asarray(result.best_x, dtype=float)
    raise KeyError("Optimizer result must contain `best_so_far_x`, `best_x`, or `x`.")


def _validate_grouping(grouping_result: Sequence[Sequence[int]]) -> None:
    """检查分组是否为空 / Validate non-empty grouping."""
    if len(grouping_result) == 0:
        raise ValueError("grouping_result must contain at least one subspace.")
    for idx, group in enumerate(grouping_result):
        if len(group) == 0:
            raise ValueError(f"grouping_result[{idx}] is empty.")


def _validate_positive_int(name: str, value: int) -> None:
    """检查正整数参数 / Validate a positive integer parameter."""
    if int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_nonnegative_int(name: str, value: int) -> None:
    """检查非负整数参数 / Validate a non-negative integer parameter."""
    if int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
