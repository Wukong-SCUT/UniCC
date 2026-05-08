from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, List, Sequence
import os
import sys

import numpy as np


# Allow running this file directly from either the project root or experiment/.
# 允许从项目根目录或 experiment/ 目录直接运行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from baseline.CMAES.cmaes import CMAES
from benchmark.cec2013lsgo.cec2013 import Benchmark
from UniCC import UniCC
from utils import evaluation_record, partition_p_and_s


@dataclass
class CEC2013ObjectiveFactory:
    """
    Pickle-safe objective factory for UniCC worker processes.
    UniCC 的子进程中使用的可序列化目标函数工厂。
    """

    fun_id: int

    def __call__(self):
        bench = Benchmark()
        return bench.get_function(self.fun_id)


def build_grouping(fun_id: int, dimension: int) -> List[List[int]]:
    """
    Build decomposition for CEC2013 LSGO functions.
    构建 CEC2013 LSGO 函数的子空间划分。
    """
    data_dir = PROJECT_ROOT / "benchmark" / "cec2013lsgo" / "datafiles"

    if fun_id in [1, 2, 3, 12, 15]:
        return [chunk.tolist() for chunk in np.array_split(np.arange(dimension), 20)]

    if fun_id in [13, 14]:
        return partition_p_and_s(
            str(data_dir / f"F{fun_id}-p.txt"),
            str(data_dir / f"F{fun_id}-s.txt"),
            overlap=5,
        )

    if fun_id in [4, 5, 6, 7]:
        grouping_result = partition_p_and_s(
            str(data_dir / f"F{fun_id}-p.txt"),
            str(data_dir / f"F{fun_id}-s_.txt"),
            overlap=0,
        )
        last_element = grouping_result[-1]
        iterator = iter(last_element)
        split_parts = [list(islice(iterator, 50)) for _ in range(0, len(last_element), 50)]
        return grouping_result[:-1] + split_parts

    return partition_p_and_s(
        str(data_dir / f"F{fun_id}-p.txt"),
        str(data_dir / f"F{fun_id}-s.txt"),
        overlap=0,
    )


def run_unicc_experiment(
    fun_id: int,
    max_fes: float,
    cycle_num: int,
    interval: int,
    pipeline_layers: int,
    window_size: int,
    seed: int,
    output_root: Path,
) -> Dict[str, Sequence]:
    """
    Run one UniCC configuration on one CEC2013 function.
    在一个 CEC2013 函数上运行一组 UniCC 参数。
    """
    bench = Benchmark()
    info = bench.get_info(fun_id)
    dimension = int(info["dimension"])
    grouping_result = build_grouping(fun_id, dimension)
    initial_best = np.zeros(dimension)

    unicc = UniCC(
        objective=None,
        objective_factory=CEC2013ObjectiveFactory(fun_id),
        optimizer_cls=CMAES,
        grouping_result=grouping_result,
        lower_boundary=info["lower"],
        upper_boundary=info["upper"],
        max_fes=max_fes,
        interval=interval,
        pipeline_layers=pipeline_layers,
        window_size=window_size,
        optimizer_options={
            "sigma": 0.5,
            "is_restart": False,
        },
    )

    print(
        f"F{fun_id}: I={interval}, L={pipeline_layers}, W={window_size}, "
        f"K_total={unicc.lag_total}, avg_K={unicc.mean_lagging_subspaces:.6f}"
    )

    result = unicc.run(
        initial_best_individual=initial_best,
        cycle_num=cycle_num,
        seed=seed + 1000 * fun_id,
        parallel_cycles=True,
    )

    data = {
        "UniCC": result.fitness_records,
        "UniCC_time": result.time_records,
    }

    output_dir = (
        output_root
        / f"budget{int(max_fes)}"
        / f"question{fun_id}"
        / f"I{interval}_L{pipeline_layers}_W{window_size}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_record(
        data,
        f"{output_dir}/",
        record_FEs_list=[1.2e5, 2e5, 1e6, 2e6, 3e6],
    )

    with open(output_dir / "schedule_params.txt", "w") as file:
        file.write(f"interval: {interval}\n")
        file.write(f"pipeline_layers: {pipeline_layers}\n")
        file.write(f"window_size: {window_size}\n")
        file.write(f"K_total: {unicc.lag_total}\n")
        file.write(f"avg_K_total: {unicc.mean_lagging_subspaces}\n")

    return data


def main() -> None:
    cycle_num = 4
    max_fes = 3e6
    seed = 42

    output_root = PROJECT_ROOT / "save_dir" / "unicc" / "cec2013lsgo"

    for fun_id in range(4, 5):
        for interval in range(5, 6):
            for pipeline_layers in range(1, 5):
                for window_size in range(1, 6):
                    run_unicc_experiment(
                        fun_id=fun_id,
                        max_fes=max_fes,
                        cycle_num=cycle_num,
                        interval=interval,
                        pipeline_layers=pipeline_layers,
                        window_size=window_size,
                        seed=seed,
                        output_root=output_root,
                    )


if __name__ == "__main__":
    # Keep BLAS libraries from oversubscribing CPU cores inside multiple processes.
    # 避免多进程中 BLAS 线程过度抢占 CPU。
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
