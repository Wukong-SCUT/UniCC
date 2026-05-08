"""
F14: Shifted Rotated Conflicting Schwefel / 位移旋转冲突 Schwefel 函数.

CEC 2013 大规模全局优化基准测试函数 F14。
CEC 2013 Large-Scale Global Optimization Benchmark Function F14.

函数特性 / Function Characteristics:
    - 多子组件分解 / Multi-subcomponent decomposition
    - 20 个子组件 / 20 sub-components
    - 冲突子空间 / Conflicting subspaces (overlap=5)
    - 独立位移向量 / Independent shift vectors for each sub-component
    - 旋转和变换 / Rotation and transforms
    - Schwefel 函数 / Schwefel function
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from .benchmarks import Benchmarks


class F14(Benchmarks):
    """
    F14: Shifted Rotated Conflicting Schwefel / 位移旋转冲突 Schwefel 函数.

    CEC 2013 基准测试函数 F14，每个子组件有独立的位移向量。
    CEC 2013 benchmark function F14 with independent shift vectors for each sub-component.

    Attributes:
        ID: 函数标识符 / Function identifier (14)
        s_size: 子问题数量 / Number of sub-problems (20)
        dimension: 问题维度 / Problem dimension (905, overlapping)
        overlap: 重叠大小 / Overlap size (5)
        minX: 搜索下界 / Lower search bound (-100.0)
        maxX: 搜索上界 / Upper search bound (100.0)
    """

    def __init__(self) -> None:
        """初始化 F14 函数 / Initialize F14 function."""
        super().__init__()
        self.ID = 14
        self.s_size = 20
        self.dimension = 905
        self.overlap = 5
        self.s = self.readS(self.s_size)
        # 注意：F14 读取的是 OvectorVec (一组向量)，不是单个 Ovector
        # Note: F14 reads OvectorVec (a set of vectors), not a single Ovector
        self.OvectorVec = self.readOvectorVec()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.w = self.readW(self.s_size)
        self.minX = -100.0
        self.maxX = 100.0

        # 预计算冲突索引和对应的位移向量 / Pre-compute conflicting indices and shift vectors
        self.sub_problems: List[Tuple[npt.NDArray, npt.NDArray, npt.NDArray, float]] = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}

        for i in range(self.s_size):
            dim = self.s[i]
            start_idx = c - i * self.overlap
            end_idx = c + dim - i * self.overlap

            indices = self.Pvector[start_idx : end_idx]
            # 获取当前子问题的独立位移向量 / Get independent shift vector for current sub-problem
            ovec_sub = self.OvectorVec[i]

            self.sub_problems.append((indices, ovec_sub, rot_map[dim], self.w[i]))
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
        计算 F14 函数值 / Compute F14 function value.

        实现逻辑 / Implementation logic:
            - F14 没有全局 Shift，是在每个子空间内独立 Shift
            - F14 has no global shift, shifts independently within each subspace
            - 切片 -> 局部位移 -> 旋转 -> 变换 -> Schwefel / Slice -> Local shift -> Rotate -> Transforms -> Schwefel
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        result = np.zeros(x.shape[0], dtype=x.dtype)

        for indices, ovec_sub, matrix, w in self.sub_problems:
            # 1. 切片原始 X / Slice raw X
            z = x[:, indices]
            # 2. 局部位移 / Local shift
            z = z - ovec_sub
            # 3. 旋转 / Rotate
            z = z @ matrix.T
            # 4. 变换 / Transforms
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            result += w * self.schwefel(z)

        return result
