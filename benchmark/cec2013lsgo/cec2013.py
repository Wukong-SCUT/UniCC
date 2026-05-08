"""
CEC 2013 基准测试入口 / CEC 2013 Benchmark Entry Point.

此模块提供统一的接口来访问所有 CEC 2013 基准测试函数。
This module provides a unified interface to access all CEC 2013 benchmark functions.

使用示例 / Usage Example:
    >>> from benchmark.cec2013lsgo.cec2013 import Benchmark
    >>> benchmark = Benchmark()
    >>> func = benchmark.get_function(11)  # 获取 F11 函数 / Get F11 function
    >>> fitness = func(np.random.uniform(-100, 100, 1000))
    >>> info = benchmark.get_info(11)
"""

from typing import Dict, Any

from .f1 import F1 as f1
from .f2 import F2 as f2
from .f3 import F3 as f3
from .f4 import F4 as f4
from .f5 import F5 as f5
from .f6 import F6 as f6
from .f7 import F7 as f7
from .f8 import F8 as f8
from .f9 import F9 as f9
from .f10 import F10 as f10
from .f11 import F11 as f11
from .f12 import F12 as f12
from .f13 import F13 as f13
from .f14 import F14 as f14
from .f15 import F15 as f15


# 函数映射表 / Function mapping table
_FUNCTION_CLASSES: Dict[int, Any] = {
    1: f1,
    2: f2,
    3: f3,
    4: f4,
    5: f5,
    6: f6,
    7: f7,
    8: f8,
    9: f9,
    10: f10,
    11: f11,
    12: f12,
    13: f13,
    14: f14,
    15: f15,
}

# 函数总数 / Total number of functions
NUM_FUNCTIONS = 15


class Benchmark:
    """
    CEC 2013 基准测试接口 / CEC 2013 Benchmark Interface.

    提供统一的接口来访问所有 CEC 2013 大规模全局优化基准测试函数。
    Provides a unified interface to access all CEC 2013 large-scale global optimization benchmark functions.

    Attributes:
        NUM_FUNCTIONS: 可用函数数量 / Number of available functions (15)

    Example:
        >>> benchmark = Benchmark()
        >>> func = benchmark.get_function(11)
        >>> fitness = func(np.random.uniform(-100, 100, 1000))
        >>> info = benchmark.get_info(11)
        >>> print(info)
        {'best': 0.0, 'dimension': 1000, 'lower': -100.0, 'upper': 100.0}
    """

    def __init__(self) -> None:
        """初始化基准测试接口 / Initialize benchmark interface."""
        self._num_functions = NUM_FUNCTIONS

    def get_function(self, func_id: int):
        """
        获取基准测试函数 / Get benchmark function.

        Args:
            func_id: 函数 ID (1-15) / Function ID (1-15)

        Returns:
            基准测试函数实例 / Benchmark function instance

        Raises:
            ValueError: 当函数 ID 超出范围时 / When function ID is out of range
        """
        if func_id not in _FUNCTION_CLASSES:
            raise ValueError(
                f"函数 ID 超出范围 / Function ID out of range: {func_id}. "
                f"有效范围 / Valid range: 1-{NUM_FUNCTIONS}"
            )
        return _FUNCTION_CLASSES[func_id]()

    def get_info(self, func_id: int) -> Dict[str, Any]:
        """
        获取函数信息 / Get function information.

        Args:
            func_id: 函数 ID (1-15) / Function ID (1-15)

        Returns:
            包含函数信息的字典 / Dictionary containing function information:
                - 'best': 最优值 / Best fitness value
                - 'dimension': 维度 / Dimension
                - 'lower': 下界 / Lower bound
                - 'upper': 上界 / Upper bound

        Raises:
            ValueError: 当函数 ID 超出范围时 / When function ID is out of range
        """
        if func_id not in _FUNCTION_CLASSES:
            raise ValueError(
                f"函数 ID 超出范围 / Function ID out of range: {func_id}. "
                f"有效范围 / Valid range: 1-{NUM_FUNCTIONS}"
            )
        func = _FUNCTION_CLASSES[func_id]()
        return func.info()

    def get_num_functions(self) -> int:
        """
        获取可用函数数量 / Get number of available functions.

        Returns:
            可用函数数量 / Number of available functions
        """
        return self._num_functions
