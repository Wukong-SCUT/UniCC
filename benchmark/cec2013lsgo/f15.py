"""
F15: Shifted Schwefel Function / 位移 Schwefel 函数.

CEC 2013 大规模全局优化基准测试函数 F15。
CEC 2013 Large-Scale Global Optimization Benchmark Function F15.

函数特性 / Function Characteristics:
    - 单组件非可分函数 / Single-component non-separable function
    - 位移和变换 / Shift and transforms
    - Schwefel 函数 / Schwefel function
"""

from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt

from .benchmarks import Benchmarks


class F15(Benchmarks):
    """
    F15: Shifted Schwefel Function / 位移 Schwefel 函数.

    CEC 2013 基准测试函数 F15，使用 Schwefel 函数。
    CEC 2013 benchmark function F15 using Schwefel function.

    Attributes:
        ID: 函数标识符 / Function identifier (15)
        minX: 搜索下界 / Lower search bound (-100.0)
        maxX: 搜索上界 / Upper search bound (100.0)
        dimension: 问题维度 / Problem dimension (1000)
    """

    def __init__(self) -> None:
        """初始化 F15 函数 / Initialize F15 function."""
        super().__init__()
        self.ID = 15
        self.dimension = 1000
        self.Ovector = self.readOvector()
        self.minX = -100.0
        self.maxX = 100.0

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
        计算 F15 函数值 / Compute F15 function value.

        逻辑: Global Shift -> OSZ -> ASY -> Schwefel
        Logic: Global Shift -> OSZ -> ASY -> Schwefel
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        z = x - self.Ovector
        z = self.transform_osz(z)
        z = self.transform_asy(z, 0.2)
        result = self.schwefel(z)

        return result
