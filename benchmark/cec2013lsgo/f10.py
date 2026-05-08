"""
F10: 20-Subcomponent Rotated Ackley / 20 子组件旋转 Ackley 函数.

CEC 2013 大规模全局优化基准测试函数 F10。
CEC 2013 Large-Scale Global Optimization Benchmark Function F10.

函数特性 / Function Characteristics:
    - 多子组件分解 / Multi-subcomponent decomposition
    - 20 个子组件 / 20 sub-components
    - 旋转和变换 / Rotation and transforms
    - Ackley 函数 / Ackley function
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from .benchmarks import Benchmarks


class F10(Benchmarks):
    """
    F10: 20-Subcomponent Rotated Ackley / 20 子组件旋转 Ackley 函数.

    CEC 2013 基准测试函数 F10，使用 Ackley 函数作为子组件。
    CEC 2013 benchmark function F10 using Ackley function as sub-components.

    Attributes:
        ID: 函数标识符 / Function identifier (10)
        s_size: 子问题数量 / Number of sub-problems (20)
        minX: 搜索下界 / Lower search bound (-32.0)
        maxX: 搜索上界 / Upper search bound (32.0)
    """

    def __init__(self) -> None:
        """初始化 F10 函数 / Initialize F10 function."""
        super().__init__()
        self.ID = 10
        self.s_size = 20
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.minX = -32.0
        self.maxX = 32.0

        # 预计算子问题配置 / Pre-compute sub-problem configuration
        self.sub_problems: List[Tuple[npt.NDArray, npt.NDArray, float]] = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}
        for i in range(self.s_size):
            dim = self.s[i]
            self.sub_problems.append((self.Pvector[c:c+dim], rot_map[dim], self.w[i]))
            c += dim

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
        计算 F10 函数值 / Compute F10 function value.

        逻辑: Global Shift -> Sub-problems (Rotate -> OSZ -> ASY -> Lambda -> Ackley)
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
            result += w * self.ackley(z)

        return result
