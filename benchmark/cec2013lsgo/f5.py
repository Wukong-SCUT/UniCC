"""
F5: 7-Subcomponent Rotated Rastrigin / 7 子组件旋转 Rastrigin 函数.

CEC 2013 大规模全局优化基准测试函数 F5。
CEC 2013 Large-Scale Global Optimization Benchmark Function F5.

函数特性 / Function Characteristics:
    - 多子组件分解 / Multi-subcomponent decomposition
    - 7 个子组件 / 7 sub-components
    - 旋转和变换 / Rotation and transforms
    - Rastrigin 函数 / Rastrigin function
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from .benchmarks import Benchmarks


class F5(Benchmarks):
    """
    F5: 7-Subcomponent Rotated Rastrigin / 7 子组件旋转 Rastrigin 函数.

    CEC 2013 基准测试函数 F5，使用 Rastrigin 函数作为子组件。
    CEC 2013 benchmark function F5 using Rastrigin function as sub-components.

    Attributes:
        ID: 函数标识符 / Function identifier (5)
        s_size: 子问题数量 / Number of sub-problems (7)
        minX: 搜索下界 / Lower search bound (-5.0)
        maxX: 搜索上界 / Upper search bound (5.0)
    """

    def __init__(self) -> None:
        """初始化 F5 函数 / Initialize F5 function."""
        super().__init__()
        self.ID = 5
        self.s_size = 7
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.minX = -5.0
        self.maxX = 5.0

        # 预计算子问题配置 / Pre-compute sub-problem configuration
        self.sub_problems: List[Tuple[npt.NDArray, npt.NDArray, float]] = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}
        for i in range(self.s_size):
            dim = self.s[i]
            self.sub_problems.append((self.Pvector[c:c+dim], rot_map[dim], self.w[i]))
            c += dim
        self.c_end = c

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
        计算 F5 函数值 / Compute F5 function value.

        逻辑: Global Shift -> Sub-problems (Rotate -> OSZ -> ASY -> Lambda -> Rastrigin)
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        result = np.zeros(x.shape[0], dtype=x.dtype)
        z_global = x - self.Ovector

        for indices, matrix, w in self.sub_problems:
            z = z_global[:, indices] @ matrix.T
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            z = self.Lambda(z, 10)
            result += w * self.rastrigin(z)

        if self.c_end < self.dimension:
            z = z_global[:, self.Pvector[self.c_end : self.dimension]]
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            z = self.Lambda(z, 10)
            result += self.rastrigin(z)

        return result
