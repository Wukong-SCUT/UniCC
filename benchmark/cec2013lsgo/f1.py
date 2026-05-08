"""
F1: Shifted Elliptic Function / 位移椭球函数.

CEC 2013 大规模全局优化基准测试函数 F1。
CEC 2013 Large-Scale Global Optimization Benchmark Function F1.

函数特性 / Function Characteristics:
    - 单组件非可分函数 / Single-component non-separable function
    - 位移变换 / Shift transformation
    - OSZ 变换 / OSZ transform
    - Elliptic 函数 / Elliptic function
"""

from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt

from .benchmarks import Benchmarks


class F1(Benchmarks):
    """
    F1: Shifted Elliptic Function / 位移椭球函数.

    CEC 2013 基准测试函数 F1，使用 Elliptic 函数。
    CEC 2013 benchmark function F1 using Elliptic function.

    Attributes:
        ID: 函数标识符 / Function identifier (1)
        minX: 搜索下界 / Lower search bound (-100.0)
        maxX: 搜索上界 / Upper search bound (100.0)
        dimension: 问题维度 / Problem dimension (1000)

    Example:
        >>> from benchmark.cec2013lsgo.f1 import F1
        >>> func = F1()
        >>> x = np.random.uniform(-100, 100, 1000)
        >>> fitness = func(x)
    """

    def __init__(self) -> None:
        """初始化 F1 函数 / Initialize F1 function."""
        super().__init__()
        self.ID = 1
        self.Ovector = self.readOvector()
        self.minX = -100.0
        self.maxX = 100.0

    def __call__(self, x: Union[npt.NDArray, List]) -> npt.NDArray:
        """
        使函数可调用 / Make function callable.

        Args:
            x: 输入向量 / Input vector

        Returns:
            适应度值 / Fitness value
        """
        return self.compute(x)

    def info(self) -> Dict[str, Union[int, float]]:
        """
        获取函数信息 / Get function information.

        Returns:
            包含函数信息的字典 / Dictionary containing function information
        """
        return {
            'best': 0.0,
            'dimension': self.dimension,
            'lower': self.minX,
            'upper': self.maxX
        }

    def compute(self, x: Union[npt.NDArray, List]) -> npt.NDArray:
        """
        计算 F1 函数值 / Compute F1 function value.

        实现逻辑 / Implementation logic:
            1. 全局位移 / Global shift
            2. OSZ 变换 / OSZ transform
            3. Elliptic 函数 / Elliptic function

        Args:
            x: 输入向量或矩阵 / Input vector or matrix

        Returns:
            适应度值 / Fitness value(s)
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        # 逻辑: Shift -> Transform -> Elliptic
        z = x - self.Ovector
        z = self.transform_osz(z)
        result = self.elliptic(z)

        return result
