"""
F2: Shifted Rastrigin Function / 位移 Rastrigin 函数.

CEC 2013 大规模全局优化基准测试函数 F2。
CEC 2013 Large-Scale Global Optimization Benchmark Function F2.

函数特性 / Function Characteristics:
    - 单组件非可分函数 / Single-component non-separable function
    - 位移和变换 / Shift and transforms
    - Rastrigin 函数 / Rastrigin function
"""

from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt

from .benchmarks import Benchmarks


class F2(Benchmarks):
    """
    F2: Shifted Rastrigin Function / 位移 Rastrigin 函数.

    CEC 2013 基准测试函数 F2，使用 Rastrigin 函数。
    CEC 2013 benchmark function F2 using Rastrigin function.

    Attributes:
        ID: 函数标识符 / Function identifier (2)
        minX: 搜索下界 / Lower search bound (-5.0)
        maxX: 搜索上界 / Upper search bound (5.0)
        dimension: 问题维度 / Problem dimension (1000)
    """

    def __init__(self) -> None:
        """初始化 F2 函数 / Initialize F2 function."""
        super().__init__()
        self.ID = 2
        self.Ovector = self.readOvector()
        self.minX = -5.0
        self.maxX = 5.0

    def __call__(self, x: Union[npt.NDArray, List]) -> npt.NDArray:
        """使函数可调用 / Make function callable."""
        return self.compute(x)

    def info(self) -> Dict[str, Union[int, float]]:
        """获取函数信息 / Get function information."""
        return {
            'best': 0.0,
            'dimension': self.dimension,
            'lower': self.minX,
            'upper': self.maxX
        }

    def compute(self, x: Union[npt.NDArray, List]) -> npt.NDArray:
        """
        计算 F2 函数值 / Compute F2 function value.

        逻辑: Shift -> OSZ -> ASY -> Lambda -> Rastrigin
        Logic: Shift -> OSZ -> ASY -> Lambda -> Rastrigin
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        z = x - self.Ovector
        z = self.transform_osz(z)
        z = self.transform_asy(z, 0.2)
        z = self.Lambda(z, 10)
        result = self.rastrigin(z)

        return result
