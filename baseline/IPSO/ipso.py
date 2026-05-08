import numpy as np  # engine for numerical computing

from baseline.IPSO.optimizer import Optimizer  # abstract class of all optimizers for continuous black-box minimization
from baseline.IPSO.pso import PSO  # particle swarm optimizer (PSO)


class IPSO(PSO):
    """Incremental Particle Swarm Optimizer (IPSO).

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'n_individuals' - swarm (population) size, aka number of particles (`int`, default: `20`),
                * 'constriction'  - constriction factor (`float`, default: `0.729`),
                * 'cognition'     - cognitive learning rate (`float`, default: `2.05`),
                * 'society'       - social learning rate (`float`, default: `2.05`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.5`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.ipso import IPSO
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> ipso = IPSO(problem, options)  # initialize the optimizer class
       >>> results = ipso.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"IPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       IPSO: 5000, 2.29225104244031e-07

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/4pk3ssrf>`_ for more details.

    Attributes
    ----------
    cognition     : `float`
                    cognitive learning rate, aka acceleration coefficient.
    constriction  : `float`
                    constriction factor.
    max_ratio_v   : `float`
                    maximal ratio of velocities w.r.t. search range.
    n_individuals : `int`
                    swarm (population) size, aka number of particles.
    society       : `float`
                    social learning rate, aka acceleration coefficient.

    References
    ----------
    De Oca, M.A.M., Stutzle, T., Van den Enden, K. and Dorigo, M., 2011.
    Incremental social learning in particle swarms.
    IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 41(2), pp.368-384.
    https://ieeexplore.ieee.org/document/5582312
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.n_individuals = 1  # minimum of swarm size
        self.max_n_individuals = options.get('max_n_individuals', 1000)  # maximum of swarm size
        assert self.max_n_individuals > 0
        self.cognition = options.get('cognition', 2.05)  # cognitive learning rate
        assert self.cognition > 0.0
        self.society = options.get('society', 2.05)  # social learning rate
        assert self.society > 0.0
        self.constriction = options.get('constriction', 0.729)  # constriction factor
        assert self.constriction > 0.0
        self.max_ratio_v = options.get('max_ratio_v', 0.5)  # maximal ratio of velocity
        assert 0.0 <= self.max_ratio_v <= 1.0

    def initialize(self, args=None):
        v = np.zeros((self.n_individuals, self.ndim_problem))  # velocities
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=self._swarm_shape)  # positions
        y = np.empty((self.n_individuals,))  # fitness
        p_x, p_y = np.copy(x), np.copy(y)  # personally previous-best positions and fitness
        
        # --- [修改开始] ---
        if self._check_terminations():
            return v, x, y, p_x, p_y
        
        # 批量评估整个初始种群
        y = self._evaluate_fitness(x, args) 
        # --- [修改结束] ---
        
        p_y = np.copy(y)
        return v, x, y, p_x, p_y
    

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, args=None, fitness=None):
        if self._check_terminations():
            return v, x, y, p_x, p_y

        # --- [关键修改 1] 获取当前的真实种群大小 ---
        # 不要使用 self.n_individuals，因为在并行调用中它可能未及时更新
        # 直接信任传入的 x 数组的长度
        n_now = x.shape[0]

        # --- 1. 现有粒子群更新 (水平社会学习 - 向量化) ---
        
        # 批量生成随机数，形状必须与当前的 x 完全一致
        cognition_rand = self.rng_optimization.uniform(size=(n_now, self.ndim_problem))
        society_rand = self.rng_optimization.uniform(size=(n_now, self.ndim_problem))
        
        # 获取全局最优位置 (g_best)
        # 这里的 argmin 会在当前长度 (n_now) 内寻找，保证安全
        g_best_idx = np.argmin(p_y)
        g_best_x = p_x[g_best_idx]
        
        # 向量化速度更新
        # v, p_x, x 的长度都是 n_now，这里不会报错
        v_new = self.constriction * (
            v + 
            self.cognition * cognition_rand * (p_x - x) +
            self.society * society_rand * (g_best_x - x)
        )
        
        # 速度限制和位置更新
        v = np.clip(v_new, self._min_v, self._max_v)
        x += v
        x = np.clip(x, self.lower_boundary, self.upper_boundary)
        
        # --- [关键修改 2] 批量评估 (并行化) ---
        # 这里的 x 长度是 n_now，所以 y_new 长度也必须是 n_now
        y_new = self._evaluate_fitness(x, args)
        
        # 强制确保 y_new 是 numpy 数组 (防止 evaluate 返回 list 导致后续比较出错)
        y_new = np.array(y_new)
        
        # 形状检查 (调试用，防止 evaluate 函数吞数据)
        if y_new.shape[0] != n_now:
            # 如果出现 21 vs 70 的错误，通常是因为 evaluate 函数实现有问题
            # 或者 x 传入时就是错的。这里我们不做处理，让它在下面报错，
            # 但上面的 n_now 逻辑通常能避免这个问题。
            pass

        # 向量化 P_best 更新
        # 此时 y_new 和 p_y 的长度都是 n_now，比较是安全的
        better_mask = y_new < p_y 
        p_x[better_mask] = x[better_mask]
        p_y[better_mask] = y_new[better_mask]
        
        # 更新当前适应度
        y = y_new 
        
        # --- 2. 增量人口增长 (垂直社会学习) ---
        
        # 只有在还没达到最大限制时才增长
        if n_now < self.max_n_individuals:
            if self._check_terminations():
                return v, x, y, p_x, p_y
            
            # 生成新粒子
            xx = self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary)
            
            # 引导向当前最好的模型
            # 注意：这里我们重新计算 argmin，因为 p_y 刚刚被更新了
            model = p_x[np.argmin(p_y)]
            xx += self.rng_optimization.uniform(size=(self.ndim_problem,)) * (model - xx)
            xx = np.clip(xx, self.lower_boundary, self.upper_boundary)
            
            # 评估新粒子 (单个评估，无需并行)
            yy = self._evaluate_fitness(xx, args)
            
            # --- [关键修改 3] 更新数组结构 ---
            # 必须使用 vstack/hstack 将新粒子拼接到数组末尾
            # 这样下一次 iterate 调用时，n_now 就会自动 +1
            
            # v 添加一行 0
            v = np.vstack((v, np.zeros((1, self.ndim_problem))))
            # x 添加一行 xx
            x = np.vstack((x, xx))
            # y 添加一个值
            y = np.hstack((y, yy))
            # p_x 添加一行 xx
            p_x = np.vstack((p_x, xx))
            # p_y 添加一个值
            p_y = np.hstack((p_y, yy))
            
            # 更新内部计数器 (仅用于记录，不用于逻辑控制)
            self.n_individuals = x.shape[0]
            
        self._n_generations += 1
        return v, x, y, p_x, p_y

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        v, x, y, p_x, p_y = self.initialize(args)
        while not self.termination_signal:
            self._print_verbose_info(fitness, y)
            v, x, y, p_x, p_y = self.iterate(v, x, y, p_x, p_y, args)
        return self._collect(fitness, y)