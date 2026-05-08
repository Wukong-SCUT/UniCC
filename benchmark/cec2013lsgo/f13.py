"""
F13: Shifted Rotated Conforming Schwefel (Overlapping) / 位移旋转重叠 Schwefel 函数.

CEC 2013 大规模全局优化基准测试函数 F13。
CEC 2013 Large-Scale Global Optimization Benchmark Function F13.

函数特性 / Function Characteristics:
    - 多子组件分解 / Multi-subcomponent decomposition
    - 20 个子组件 / 20 sub-components
    - 重叠子空间 / Overlapping subspaces (overlap=5)
    - 旋转和变换 / Rotation and transforms
    - Schwefel 函数 / Schwefel function
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from .benchmarks import Benchmarks


class F13(Benchmarks):
    """
    F13: Shifted Rotated Conforming Schwefel (Overlapping) / 位移旋转重叠 Schwefel 函数.

    CEC 2013 基准测试函数 F13，具有重叠子空间的特性。
    CEC 2013 benchmark function F13 with overlapping subspace characteristic.

    Attributes:
        ID: 函数标识符 / Function identifier (13)
        s_size: 子问题数量 / Number of sub-problems (20)
        dimension: 问题维度 / Problem dimension (905, overlapping)
        overlap: 重叠大小 / Overlap size (5)
        minX: 搜索下界 / Lower search bound (-100.0)
        maxX: 搜索上界 / Upper search bound (100.0)
    """

    def __init__(self) -> None:
        """初始化 F13 函数 / Initialize F13 function."""
        super().__init__()
        self.ID = 13
        self.s_size = 20
        self.dimension = 905  # 重叠维度 / Overlapping dimensions
        self.overlap = 5
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.minX = -100.0
        self.maxX = 100.0

        # 预计算重叠索引 / Pre-compute overlapping indices
        self.sub_problems: List[Tuple[npt.NDArray, npt.NDArray, float]] = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}

        for i in range(self.s_size):
            dim = self.s[i]
            # 计算重叠后的实际索引位置 / Calculate actual index positions with overlap
            start_idx = c - i * self.overlap
            end_idx = c + dim - i * self.overlap

            indices = self.Pvector[start_idx : end_idx]

            self.sub_problems.append((indices, rot_map[dim], self.w[i]))
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
        计算 F13 函数值 / Compute F13 function value.

        实现逻辑 / Implementation logic:
            1. 全局位移 / Global shift
            2. 子问题循环（含重叠）/ Sub-problem loop (with overlap)
            3. 切片 -> 旋转 -> OSZ -> ASY -> Schwefel / Slice -> Rotate -> OSZ -> ASY -> Schwefel
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        result = np.zeros(x.shape[0], dtype=x.dtype)

        # 全局位移 / Global shift
        z_global = x - self.Ovector

        for indices, matrix, w in self.sub_problems:
            # 切片（重叠）-> 旋转 -> OSZ -> ASY -> Schwefel
            # Slice (Overlapping) -> Rotate -> OSZ -> ASY -> Schwefel
            z = z_global[:, indices] @ matrix.T
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            result += w * self.schwefel(z)

        return result
