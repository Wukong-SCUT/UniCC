"""
CEC 2013 大规模全局优化基准测试 / CEC 2013 Large-Scale Global Optimization Benchmark.

此模块实现了 CEC 2013 竞赛的大规模全局优化基准测试函数。
This module implements the CEC 2013 Large-Scale Global Optimization benchmark functions.

主要组件 / Main Components:
    - Benchmarks: 基准测试基类，提供核心功能 / Base benchmark class providing core functionality
    - JIT Functions: Numba 加速的核心测试函数 / Numba-accelerated core test functions

基准函数 / Benchmark Functions:
    - Sphere, Elliptic, Rastrigin, Ackley, Schwefel, Rosenbrock

变换函数 / Transform Functions:
    - OSZ 变换 / OSZ transform
    - ASY 变换 / ASY transform
    - Lambda 变换 / Lambda transform

参考 / References:
    - Xiaodong Li, et al. "Benchmark Functions for CEC'2013 Special Session
      and Competition on Large-Scale Global Optimization." 2013.

使用示例 / Usage Example:
    >>> from benchmark.cec2013lsgo.f11 import F11
    >>> func = F11()
    >>> fitness = func(np.random.uniform(-100, 100, 1000))
"""

import logging
from typing import List, Optional, Tuple, Union

import math
import numpy as np
import numpy.typing as npt
from numba import njit

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# 常量定义 / Constants
# ==========================================

# 默认数据目录 / Default data directory
DEFAULT_DATA_DIR = "./benchmark/cec2013lsgo/datafiles"

# 默认维度 / Default dimensions
DEFAULT_DIMENSION = 1000
MIN_SUB_DIM = 25
MED_SUB_DIM = 50
MAX_SUB_DIM = 100

# 默认评估次数 / Default evaluation counts
DEFAULT_MAX_EVALS = 3000000

# 默认子问题数量 / Default number of sub-problems
DEFAULT_S_SIZE = 20

# ==========================================
# Numba JIT 加速核心函数 / Numba JIT Accelerated Core Functions
# ==========================================


@njit(fastmath=True, cache=True)
def jit_transform_osz(z: npt.NDArray) -> npt.NDArray:
    """
    OSZ 变换 / OSZ (Oscillation) Transform.

    实现振荡变换，用于增加目标函数的复杂性。
    Implements the oscillation transform to increase objective function complexity.

    Args:
        z: 输入数组 / Input array

    Returns:
        变换后的数组 / Transformed array (in-place modified)
    """
    flat_z = z.ravel()
    n = flat_z.size

    for i in range(n):
        val = flat_z[i]
        if val == 0:
            continue

        abs_val = math.fabs(val)
        hat = math.log(abs_val)

        if val > 0:
            c1, c2 = 10.0, 7.9
        else:
            c1, c2 = 5.5, 3.1

        sin_term = math.sin(c1 * hat) + math.sin(c2 * hat)

        # 代数化简优化 / Algebraic simplification optimization
        flat_z[i] = val * math.exp(0.049 * sin_term)

    return z


@njit(fastmath=True, cache=True)
def jit_transform_asy(z: npt.NDArray, beta: float) -> npt.NDArray:
    """
    ASY 变换 / ASY (Asymmetry) Transform.

    实现非对称变换，破坏目标函数的对称性。
    Implements the asymmetry transform to break objective function symmetry.

    Args:
        z: 输入数组 / Input array, shape (n, dim)
        beta: 非对称参数 / Asymmetry parameter

    Returns:
        变换后的数组 / Transformed array
    """
    n, dim = z.shape
    for i in range(n):
        for j in range(dim):
            val = z[i, j]
            if val > 0:
                sqrt_val = math.sqrt(val)
                exponent = 1 + beta * (j / (dim - 1)) * sqrt_val
                z[i, j] = val ** exponent
    return z


@njit(fastmath=True, cache=True)
def jit_lambda(z: npt.NDArray, alpha: float) -> npt.NDArray:
    """
    Lambda 变换 / Lambda Transform.

    实现缩放变换，用于调整搜索空间的形状。
    Implements scaling transform to adjust search space shape.

    Args:
        z: 输入数组 / Input array, shape (n, dim)
        alpha: 缩放因子 / Scaling factor

    Returns:
        变换后的数组 / Transformed array
    """
    n, dim = z.shape
    for i in range(n):
        for j in range(dim):
            exponent = 0.5 * j / (dim - 1)
            factor = math.pow(alpha, exponent)
            z[i, j] *= factor
    return z


# ==========================================
# 基础基准函数 (JIT) / Basic Benchmark Functions (JIT)
# ==========================================


@njit(fastmath=True, cache=True)
def jit_sphere(x: npt.NDArray) -> npt.NDArray:
    """
    Sphere 函数 / Sphere Function.

    f(x) = sum(x_i^2)

    Args:
        x: 输入数组 / Input array, shape (n, dim)

    Returns:
        函数值 / Function values, shape (n,)
    """
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        sum_sq = 0.0
        for j in range(dim):
            val = x[i, j]
            sum_sq += val * val
        res[i] = sum_sq
    return res


@njit(fastmath=True, cache=True)
def jit_elliptic(x: npt.NDArray) -> npt.NDArray:
    """
    Elliptic 函数 / Elliptic Function.

    条件 ill-conditioned 的椭球函数。
    Ill-conditioned elliptic function.

    Args:
        x: 输入数组 / Input array, shape (n, dim)

    Returns:
        函数值 / Function values, shape (n,)
    """
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        total = 0.0
        for j in range(dim):
            coeff = math.pow(10.0, 6.0 * j / (dim - 1))
            val = x[i, j]
            total += coeff * (val * val)
        res[i] = total
    return res


@njit(fastmath=True, cache=True)
def jit_rastrigin(x: npt.NDArray) -> npt.NDArray:
    """
    Rastrigin 函数 / Rastrigin Function.

    多模态函数，包含大量局部最优。
    Multimodal function with many local optima.

    Args:
        x: 输入数组 / Input array, shape (n, dim)

    Returns:
        函数值 / Function values, shape (n,)
    """
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        total = 0.0
        for j in range(dim):
            val = x[i, j]
            total += val * val - 10.0 * math.cos(2.0 * math.pi * val) + 10.0
        res[i] = total
    return res


@njit(fastmath=True, cache=True)
def jit_ackley(x: npt.NDArray) -> npt.NDArray:
    """
    Ackley 函数 / Ackley Function.

    多模态函数，具有指数项。
    Multimodal function with exponential terms.

    Args:
        x: 输入数组 / Input array, shape (n, dim)

    Returns:
        函数值 / Function values, shape (n,)
    """
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        sum_sq = 0.0
        sum_cos = 0.0
        for j in range(dim):
            val = x[i, j]
            sum_sq += val * val
            sum_cos += math.cos(2.0 * math.pi * val)

        term1 = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / dim))
        term2 = -math.exp(sum_cos / dim)
        res[i] = term1 + term2 + 20.0 + math.e
    return res


@njit(fastmath=True, cache=True)
def jit_schwefel(x: npt.NDArray) -> npt.NDArray:
    """
    Schwefel 函数 / Schwefel Function.

    使用累积和的 Schwefel 问题。
    Schwefel problem using cumulative sums.

    Args:
        x: 输入数组 / Input array, shape (n, dim)

    Returns:
        函数值 / Function values, shape (n,)
    """
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        cumsum = 0.0
        sum_sq = 0.0
        for j in range(dim):
            cumsum += x[i, j]
            sum_sq += cumsum * cumsum
        res[i] = sum_sq
    return res


@njit(fastmath=True, cache=True)
def jit_rosenbrock(x: npt.NDArray) -> npt.NDArray:
    """
    Rosenbrock 函数 / Rosenbrock Function.

    经典的优化测试函数，具有狭长的山谷。
    Classic optimization test function with a narrow valley.

    Args:
        x: 输入数组 / Input array, shape (n, dim)

    Returns:
        函数值 / Function values, shape (n,)
    """
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        total = 0.0
        for j in range(dim - 1):
            x0 = x[i, j]
            x1 = x[i, j + 1]
            t = x0 * x0 - x1
            term1 = 100.0 * t * t
            term2 = (x0 - 1.0) * (x0 - 1.0)
            total += term1 + term2
        res[i] = total
    return res


# ==========================================
# 主类 / Main Class
# ==========================================


class Benchmarks:
    """
    CEC 2013 基准测试基类 / Base Class for CEC 2013 Benchmark Functions.

    提供基准测试函数的通用功能，包括数据加载、变换和旋转。
    Provides common functionality for benchmark functions, including data loading, transforms, and rotation.

    Attributes:
        dtype: 数据类型 / Data type (default: float64)
        data_dir: 数据文件目录 / Data file directory
        dimension: 问题维度 / Problem dimension
        min_dim/med_dim/max_dim: 子问题维度 / Sub-problem dimensions
        ID: 函数 ID / Function ID
        s_size: 子问题数量 / Number of sub-problems
        maxevals: 最大评估次数 / Maximum evaluations

    Example:
        >>> from benchmark.cec2013lsgo.f11 import F11
        >>> func = F11()
        >>> fitness = func(np.random.uniform(-100, 100, 1000))
    """

    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        data_dir: str = DEFAULT_DATA_DIR
    ) -> None:
        """
        初始化基准测试 / Initialize benchmark.

        Args:
            dimension: 问题维度 / Problem dimension
            data_dir: 数据文件目录 / Data file directory
        """
        # 数据类型和维度 / Data type and dimension
        self.dtype = np.float64
        self.data_dir = data_dir
        self.dimension = dimension

        # 子问题维度 / Sub-problem dimensions
        self.min_dim = MIN_SUB_DIM
        self.med_dim = MED_SUB_DIM
        self.max_dim = MAX_SUB_DIM

        # 函数标识 / Function identification
        self.ID: Optional[int] = None
        self.s_size = DEFAULT_S_SIZE
        self.overlap: Optional[int] = None
        self.minX: Optional[float] = None
        self.maxX: Optional[float] = None

        # 向量数据 / Vector data
        self.Ovector: Optional[npt.NDArray] = None
        self.OvectorVec: Optional[List[npt.NDArray]] = None
        self.Pvector: Optional[npt.NDArray] = None

        # 旋转矩阵 / Rotation matrices
        self.r25: Optional[npt.NDArray] = None
        self.r50: Optional[npt.NDArray] = None
        self.r100: Optional[npt.NDArray] = None
        self.r_min_dim = self.min_dim
        self.r_med_dim = self.med_dim
        self.r_max_dim = self.max_dim

        # 工作数组 / Working arrays
        self.anotherz = np.zeros(self.dimension, dtype=self.dtype)
        self.anotherz1: Optional[npt.NDArray] = None

        # 评估计数 / Evaluation counting
        self.best_fitness = float('inf')
        self.maxevals = DEFAULT_MAX_EVALS
        self.numevals = 0

        # 输出配置 / Output configuration
        self.output = ""
        self.output_dir = 'cec2013lsgo_py'
        self.record_evels = [120000, 600000, 3000000]

    def readOvector(self) -> npt.NDArray:
        """
        读取最优解向量 / Read optimal solution vector.

        Returns:
            最优解向量 / Optimal solution vector

        Raises:
            FileNotFoundError: 当数据文件不存在时 / When data file doesn't exist
        """
        d = np.zeros(self.dimension, dtype=self.dtype)
        file_path = f"{self.data_dir}/F{self.ID}-xopt.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dimension:
                            d[c] = float(value)
                            c += 1
        except FileNotFoundError as e:
            logger.error(f"无法打开数据文件 / Cannot open datafile: {file_path}")
            raise FileNotFoundError(f"Cannot open datafile '{file_path}'") from e
        return d

    def readOvectorVec(self) -> List[npt.NDArray]:
        """
        读取子问题最优解向量 / Read sub-problem optimal solution vectors.

        Returns:
            子问题最优解向量列表 / List of sub-problem optimal solution vectors
        """
        d = [np.zeros(self.s[i], dtype=self.dtype) for i in range(self.s_size)]
        file_path = f"{self.data_dir}/F{self.ID}-xopt.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                i = -1
                up = 0
                for line in file:
                    if c == up:
                        i += 1
                        if i < self.s_size:
                            up += self.s[i]
                    values = line.strip().split(',')
                    for value in values:
                        if i < self.s_size and c < self.dimension:
                            idx_in_group = c - (up - self.s[i])
                            if idx_in_group < len(d[i]):
                                d[i][idx_in_group] = float(value)
                        c += 1
        except FileNotFoundError as e:
            logger.error(f"无法打开 OvectorVec 数据文件 / Cannot open OvectorVec datafile: {file_path}")
            raise FileNotFoundError(f"Cannot open OvectorVec datafile '{file_path}'") from e
        return d

    def readPermVector(self) -> npt.NDArray:
        """
        读取排列向量 / Read permutation vector.

        Returns:
            排列向量 / Permutation vector (0-indexed)

        Raises:
            FileNotFoundError: 当数据文件不存在时 / When data file doesn't exist
        """
        d = np.zeros(self.dimension, dtype=int)
        file_path = f"{self.data_dir}/F{self.ID}-p.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dimension:
                            d[c] = int(float(value)) - 1
                            c += 1
        except FileNotFoundError as e:
            logger.error(f"无法打开数据文件 / Cannot open datafile: {file_path}")
            raise FileNotFoundError(f"Cannot open datafile '{file_path}'") from e
        return d

    def readR(self, sub_dim: int) -> npt.NDArray:
        """
        读取旋转矩阵 / Read rotation matrix.

        Args:
            sub_dim: 子维度 / Sub-dimension

        Returns:
            旋转矩阵 / Rotation matrix, shape (sub_dim, sub_dim)

        Raises:
            FileNotFoundError: 当数据文件不存在时 / When data file doesn't exist
        """
        m = np.zeros((sub_dim, sub_dim), dtype=self.dtype)
        file_path = f"{self.data_dir}/F{self.ID}-R{sub_dim}.txt"
        try:
            with open(file_path, 'r') as file:
                i = 0
                for line in file:
                    values = line.strip().split(',')
                    for j, value in enumerate(values):
                        if i < sub_dim and j < sub_dim:
                            m[i, j] = float(value)
                    i += 1
        except FileNotFoundError as e:
            logger.error(f"无法打开数据文件 / Cannot open datafile: {file_path}")
            raise FileNotFoundError(f"Cannot open datafile '{file_path}'") from e
        return m

    def readS(self, num: int) -> npt.NDArray:
        """
        读取子问题维度 / Read sub-problem dimensions.

        Args:
            num: 子问题数量 / Number of sub-problems

        Returns:
            子问题维度数组 / Array of sub-problem dimensions
        """
        self.s = np.zeros(num, dtype=int)
        file_path = f"{self.data_dir}/F{self.ID}-s.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    if c < num:
                        self.s[c] = int(float(line.strip()))
                        c += 1
        except FileNotFoundError as e:
            logger.error(f"无法打开数据文件 / Cannot open datafile: {file_path}")
            raise FileNotFoundError(f"Cannot open datafile '{file_path}'") from e
        return self.s

    def readW(self, num: int) -> npt.NDArray:
        """
        读取权重向量 / Read weight vector.

        Args:
            num: 权重数量 / Number of weights

        Returns:
            权重向量 / Weight vector
        """
        self.w = np.zeros(num, dtype=self.dtype)
        file_path = f"{self.data_dir}/F{self.ID}-w.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    if c < num:
                        self.w[c] = float(line.strip())
                        c += 1
        except FileNotFoundError as e:
            logger.error(f"无法打开数据文件 / Cannot open datafile: {file_path}")
            raise FileNotFoundError(f"Cannot open datafile '{file_path}'") from e
        return self.w

    def multiply(
        self,
        vector: npt.NDArray,
        matrix: npt.NDArray
    ) -> npt.NDArray:
        """
        批量矩阵乘法 / Batch matrix multiplication.

        使用 NumPy 的 BLAS 优化实现。
        Uses NumPy's BLAS-optimized implementation.

        Args:
            vector: 向量或矩阵 / Vector or matrix
            matrix: 矩阵 / Matrix

        Returns:
            乘积结果 / Product result
        """
        if vector.ndim == 1:
            return np.dot(matrix, vector)
        return vector @ matrix.T

    def rotateVector(self, i: int, c: int) -> Optional[npt.NDArray]:
        """
        旋转向量（兼容性保留） / Rotate vector (compatibility保留).

        兼容性保留的旋转函数，主要逻辑应在子类中优化。
        Rotation function kept for compatibility; main logic should be optimized in subclasses.

        Args:
            i: 子问题索引 / Sub-problem index
            c: 起始位置 / Starting position

        Returns:
            旋转后的向量 / Rotated vector
        """
        sub_dim = self.s[i]
        indices = self.Pvector[c:c + sub_dim]

        if self.anotherz.ndim == 1:
            z = self.anotherz[indices]
        else:
            z = self.anotherz[:, indices]

        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            self.anotherz1 = None
        return self.anotherz1

    def rotateVectorConform(self, i: int, c: int) -> Optional[npt.NDArray]:
        """
        旋转向量（符合重叠） / Rotate vector with overlap conform.

        Args:
            i: 子问题索引 / Sub-problem index
            c: 起始位置 / Starting position

        Returns:
            旋转后的向量 / Rotated vector
        """
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap
        indices = self.Pvector[start_index:end_index]

        if self.anotherz.ndim == 1:
            z = self.anotherz[indices]
        else:
            z = self.anotherz[:, indices]

        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            self.anotherz1 = None
        return self.anotherz1

    def rotateVectorConflict(
        self,
        i: int,
        c: int,
        x: Union[npt.NDArray, List]
    ) -> Optional[npt.NDArray]:
        """
        旋转向量（冲突处理） / Rotate vector with conflict handling.

        Args:
            i: 子问题索引 / Sub-problem index
            c: 起始位置 / Starting position
            x: 输入向量 / Input vector

        Returns:
            旋转后的向量 / Rotated vector
        """
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap
        indices = self.Pvector[start_index:end_index]

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim == 1:
            z = x[indices] - self.OvectorVec[i]
        else:
            z = x[:, indices] - self.OvectorVec[i]

        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            self.anotherz1 = None
        return self.anotherz1

    # ==========================================
    # JIT 函数包装器 / JIT Function Wrappers
    # ==========================================

    def sphere(self, x: npt.NDArray) -> npt.NDArray:
        """Sphere 函数 / Sphere function wrapper."""
        return jit_sphere(x)

    def elliptic(self, x: npt.NDArray) -> npt.NDArray:
        """Elliptic 函数 / Elliptic function wrapper."""
        return jit_elliptic(x)

    def rastrigin(self, x: npt.NDArray) -> npt.NDArray:
        """Rastrigin 函数 / Rastrigin function wrapper."""
        return jit_rastrigin(x)

    def ackley(self, x: npt.NDArray) -> npt.NDArray:
        """Ackley 函数 / Ackley function wrapper."""
        return jit_ackley(x)

    def schwefel(self, x: npt.NDArray) -> npt.NDArray:
        """Schwefel 函数 / Schwefel function wrapper."""
        return jit_schwefel(x)

    def rosenbrock(self, x: npt.NDArray) -> npt.NDArray:
        """Rosenbrock 函数 / Rosenbrock function wrapper."""
        return jit_rosenbrock(x)

    def transform_osz(self, z: npt.NDArray) -> npt.NDArray:
        """OSZ 变换 / OSZ transform wrapper."""
        return jit_transform_osz(z)

    def transform_asy(self, z: npt.NDArray, beta: float = 0.2) -> npt.NDArray:
        """ASY 变换 / ASY transform wrapper."""
        return jit_transform_asy(z, beta)

    def Lambda(self, z: npt.NDArray, alpha: float = 10) -> npt.NDArray:
        """Lambda 变换 / Lambda transform wrapper."""
        return jit_lambda(z, alpha)
