"""
F11: Shifted Rotated Schwefel 函数 / Shifted Rotated Schwefel Function.

CEC 2013 大规模全局优化基准测试函数 F11。
CEC 2013 Large-Scale Global Optimization Benchmark Function F11.

函数特性 / Function Characteristics:
    - 多子问题分解 / Multi-subproblem decomposition
    - 旋转和位移 / Rotation and shifting
    - OSZ 和 ASY 变换 / OSZ and ASY transforms
    - 20 个子组件 / 20 sub-components

参考 / Reference:
    - CEC 2013 Competition on Large-Scale Global Optimization
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from .benchmarks import Benchmarks


class F11(Benchmarks):
    """
    F11: Shifted Rotated Schwefel 函数 / Shifted Rotated Schwefel Function.

    CEC 2013 基准测试函数 F11，使用 Schwefel 函数作为子组件。
    CEC 2013 benchmark function F11 using Schwefel function as sub-components.

    Attributes:
        ID: 函数标识符 / Function identifier (11)
        s_size: 子问题数量 / Number of sub-problems (20)
        minX: 搜索下界 / Lower search bound (-100.0)
        maxX: 搜索上界 / Upper search bound (100.0)
        dimension: 问题维度 / Problem dimension (1000)

    Example:
        >>> from benchmark.cec2013lsgo.f11 import F11
        >>> func = F11()
        >>> x = np.random.uniform(-100, 100, 1000)
        >>> fitness = func(x)
    """

    def __init__(self) -> None:
        """初始化 F11 函数 / Initialize F11 function."""
        super().__init__()
        self.ID = 11
        self.s_size = 20
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.minX = -100.0
        self.maxX = 100.0

        # 预计算子问题配置 / Pre-compute sub-problem configuration
        self.sub_problems: List[Tuple[npt.NDArray, npt.NDArray, float]] = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}
        for i in range(self.s_size):
            dim = self.s[i]
            # 预存: (切片索引, 旋转矩阵, 权重) / Pre-store: (slice indices, rotation matrix, weight)
            self.sub_problems.append((self.Pvector[c:c+dim], rot_map[dim], self.w[i]))
            c += dim

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
            包含函数信息的字典 / Dictionary containing function information:
                - 'best': 最优适应度值 / Best fitness value (0.0)
                - 'dimension': 问题维度 / Problem dimension (1000)
                - 'lower': 搜索下界 / Lower search bound (-100.0)
                - 'upper': 搜索上界 / Upper search bound (100.0)
        """
        return {
            'best': 0.0,
            'dimension': self.dimension,
            'lower': self.minX,
            'upper': self.maxX
        }

    def compute(self, x: Union[npt.NDArray, List]) -> npt.NDArray:
        """
        计算 F11 函数值 / Compute F11 function value.

        实现逻辑：
        Implementation logic:
            1. 全局位移 / Global shift
            2. 对每个子问题 / For each sub-problem:
               - 切片 / Slice
               - 旋转 / Rotate
               - OSZ 变换 / OSZ transform
               - ASY 变换 / ASY transform
               - Schwefel 函数 / Schwefel function
            3. 加权求和 / Weighted sum

        Args:
            x: 输入向量或矩阵 / Input vector or matrix

        Returns:
            适应度值 / Fitness value(s)
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        result = np.zeros(x.shape[0], dtype=x.dtype)

        # 全局位移 / Global shift
        z_global = x - self.Ovector

        # 遍历子问题 / Iterate over sub-problems
        for indices, matrix, w in self.sub_problems:
            # 切片 -> 旋转 -> OSZ -> ASY -> Schwefel
            # Slice -> Rotate -> OSZ -> ASY -> Schwefel
            z = z_global[:, indices] @ matrix.T
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            result += w * self.schwefel(z)

        return result
