# UniCC

[English](README.md) | 中文说明

UniCC 是一个面向大规模全局优化（Large-Scale Global Optimization, LSGO）的统一协同进化框架。本仓库包含可复用的 UniCC 框架、CEC2013 LSGO benchmark、若干 baseline 优化器，以及一个基于 UniCC 运行 CEC2013 LSGO 的实验脚本。

![UniCC 框架动图](assets/unicc_framework.gif)

如果希望交互式理解调度矩阵、流水线执行、加速比和信息滞后，可以打开 [UniCC.html](UniCC.html)。此前端也可以通过匿名页面访问：<https://anonymous.4open.science/w/UniCC-E7EF/UniCC.html>。

## UniCC 做什么

UniCC 将高维优化问题分解为多个子空间，并通过流水线方式调度各个子空间优化器。

- `M`：子空间数量。
- `I`：调度矩阵中相邻流水线层之间的 interval。
- `L`：流水线层数。
- `W`：窗口大小，即每个 batch 中处理多少个调度列。
- `A`：调度矩阵，用于决定子空间、流水线层和执行顺序之间的对应关系。
- `lag_total` 和 `mean_lagging_subspaces`：用于估计调度过程中跨子空间信息滞后的指标。

[UniCC.py](UniCC.py) 中的框架不绑定任何具体 benchmark 或实验设置。具体实验需要在类外提供目标函数、优化器、子空间划分、变量边界、随机种子和结果保存逻辑。

## 仓库结构

```text
.
├── UniCC.py                     # 可复用的 UniCC 框架
├── UniCC.html                   # 交互式架构可视化页面
├── utils.py                     # 目标函数包装、分组、记录和绘图工具
├── experiment/
│   └── cec2013lsgo.py           # 调用 UniCC 的 CEC2013 LSGO 实验示例
├── benchmark/cec2013lsgo/       # CEC2013 LSGO benchmark 实现与数据文件
├── baseline/                    # baseline 优化器
├── paper/                       # 论文 PDF
└── assets/
    └── unicc_framework.gif      # README 中使用的框架动图
```

`save_dir/` 已加入 `.gitignore`。实验结果默认写入该目录，但不会同步到 Git 仓库。

## 安装

建议使用 Python 3.9 或更高版本，并创建虚拟环境。

```bash
git clone git@github.com:Wukong-SCUT/UniCC.git
cd UniCC

python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib numba
```

## 运行 CEC2013 LSGO 示例实验

```bash
python experiment/cec2013lsgo.py
```

默认情况下，脚本会运行 `main()` 中设置的 UniCC 参数组合，并将结果写入：

```text
save_dir/unicc/cec2013lsgo/budget3000000/question{fun_id}/I{interval}_L{pipeline_layers}_W{window_size}/
```

每个输出目录包含：

- `evaluation_record.txt`：在若干函数评价预算点上的最优 fitness 记录。
- `schedule_params.txt`：UniCC 调度参数与滞后指标。

如需调整实验范围，可以修改 `experiment/cec2013lsgo.py` 中的循环：

```python
for fun_id in range(4, 5):
    for interval in range(5, 6):
        for pipeline_layers in range(1, 5):
            for window_size in range(1, 6):
                ...
```

## 在自己的实验中调用 UniCC

UniCC 框架只需要实验侧提供四类信息：完整空间目标函数、优化器类、子空间划分和变量边界。

```python
import numpy as np

from UniCC import UniCC
from baseline.CMAES.cmaes import CMAES


def objective(x_batch):
    return np.sum(x_batch ** 2, axis=1)


dimension = 100
grouping_result = [group.tolist() for group in np.array_split(np.arange(dimension), 10)]

unicc = UniCC(
    objective=objective,
    optimizer_cls=CMAES,
    grouping_result=grouping_result,
    lower_boundary=-100,
    upper_boundary=100,
    max_fes=100000,
    interval=1,
    pipeline_layers=2,
    window_size=3,
    optimizer_options={
        "sigma": 0.5,
        "is_restart": False,
    },
)

result = unicc.run(
    initial_best_individual=np.zeros(dimension),
    cycle_num=1,
    seed=42,
)

print(result.fitness_records)
print(result.time_records)
print(result.best_individuals[0])
```

## API 说明

`UniCC` 要求优化器类满足如下接口：

```python
optimizer = optimizer_cls(problem, options)
result = optimizer.optimize()
```

`problem` 包含：

- `fitness_function`：经过包装后的子空间目标函数。
- `ndim_problem`：子空间维度。
- `lower_boundary`：子空间下界。
- `upper_boundary`：子空间上界。

`result` 中需要包含 `best_so_far_x`、`best_x` 或 `x` 中的任意一个字段。

如果目标函数无法被多进程序列化，可以令 `objective=None`，并提供 `objective_factory`，让每个 worker 在子进程内部创建目标函数。具体示例见 [experiment/cec2013lsgo.py](experiment/cec2013lsgo.py) 中的 `CEC2013ObjectiveFactory`。

## UniCC 返回结果

`unicc.run(...)` 返回 `UniCCResult` 对象：

- `fitness_records`：每次独立运行的 fitness 历史。
- `time_records`：每次独立运行的耗时。
- `best_individuals`：每次独立运行结束后的完整空间最优个体。

常用调度属性：

```python
unicc.schedule_matrix
unicc.lag_total
unicc.mean_lagging_subspaces
```

## 说明

- `UniCC.py` 是可复用的 UniCC 框架。
- `experiment/cec2013lsgo.py` 是基于该框架的实验示例。
- `save_dir/` 下的实验记录不会被 Git 跟踪。
- 运行仓库提供的 CEC2013 LSGO 实验时，需要保留 `benchmark/cec2013lsgo/datafiles/` 中的数据文件。
