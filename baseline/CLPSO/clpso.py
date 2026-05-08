import numpy as np  # engine for numerical computing

from baseline.CLPSO.pso import PSO  # particle swarm optimizer (PSO)


class CLPSO(PSO):
    """Comprehensive Learning Particle Swarm Optimizer (CLPSO).

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
                * 'c'             - comprehensive learning rate (`float`, default: `1.49445`),
                * 'm'             - refreshing gap (`int`, default: `7`),
                * 'max_ratio_v'   - maximal ratio of velocities w.r.t. search range (`float`, default: `0.2`).

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.pso.clpso import CLPSO
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> clpso = CLPSO(problem, options)  # initialize the optimizer class
       >>> results = clpso.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"CLPSO: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       CLPSO: 5000, 7.184727085112434e-05

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/f3pp4nfh>`_ for more details.

    Attributes
    ----------
    c             : `float`
                    comprehensive learning rate.
    m             : `int`
                    refreshing gap.
    max_ratio_v   : `float`
                    maximal ratio of velocities w.r.t. search range.
    n_individuals : `int`
                    swarm (population) size, aka number of particles.

    References
    ----------
    Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006.
    Comprehensive learning particle swarm optimizer for global optimization of multimodal functions.
    IEEE Transactions on Evolutionary Computation, 10(3), pp.281-295.
    https://ieeexplore.ieee.org/abstract/document/1637688

    See the original MATLAB source code from Prof. Suganthan:
    https://github.com/P-N-Suganthan/CODES/blob/master/2006-IEEE-TEC-CLPSO.zip
    """
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.c = options.get('c', 1.49445)  # comprehensive learning rate
        assert self.c > 0.0
        self.m = options.get('m', 7)  # refreshing gap
        assert self.m > 0
        pc = 5.0*np.linspace(0, 1, self.n_individuals)
        self._pc = 0.5*(np.exp(pc) - np.exp(pc[0]))/(np.exp(pc[-1]) - np.exp(pc[0]))
        # set number of successive generations each particle has not improved its best fitness
        self._flag = np.zeros((self.n_individuals,))
        # set linearly decreasing inertia weights from 0.9 to 0.2
        self._w = 0.9 - 0.7*(np.arange(self._max_generations) + 1.0)/self._max_generations

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        N, D = self.n_individuals, self.ndim_problem
        
        if self._check_terminations():
            return v, x, y, p_x, p_y, n_x
        
        # -----------------------------------------------------------------
        # 1. 批量化拓扑学习 (Comprehensive Learning Target Selection)
        # -----------------------------------------------------------------
        
        mask_refresh = self._flag >= self.m
        
        if np.any(mask_refresh):
            idx_refresh = np.where(mask_refresh)[0]
            
            # 维度学习概率判断 (Learning Probability)
            rand_pc_mat = self.rng_optimization.random((len(idx_refresh), D))
            mask_learn = rand_pc_mat < self._pc[idx_refresh, np.newaxis] 
            
            # 向量化锦标赛选择 (Tournament Selection)
            cand_a = self.rng_optimization.integers(0, N, (len(idx_refresh), D))
            offset = self.rng_optimization.integers(1, N, (len(idx_refresh), D))
            cand_b = (cand_a + offset) % N
            
            winners = np.where(p_y[cand_a] < p_y[cand_b], cand_a, cand_b)
            
            # 初始的 exemplar_indices 矩阵，默认为自身索引
            exemplar_indices = np.tile(idx_refresh[:, None], (1, D))
            exemplar_indices = np.where(mask_learn, winners, exemplar_indices)
            
            # 处理 “All Self” 边缘情况
            is_all_self = np.all(exemplar_indices == idx_refresh[:, None], axis=1)
            
            if np.any(is_all_self):
                sub_idx_all_self = np.where(is_all_self)[0]
                rand_dims = self.rng_optimization.integers(0, D, size=len(sub_idx_all_self))
                offsets = self.rng_optimization.integers(1, N, size=len(sub_idx_all_self))
                real_indices = idx_refresh[sub_idx_all_self]
                new_partners = (real_indices + offsets) % N
                exemplar_indices[sub_idx_all_self, rand_dims] = new_partners
            
            # 更新 n_x (目标位置)
            col_indices = np.arange(D)
            n_x[mask_refresh] = p_x[exemplar_indices, col_indices]
            
            self._flag[mask_refresh] = 0

        # -----------------------------------------------------------------
        # 2. 批量化速度和位置更新 (Velocity and Position Update)
        # -----------------------------------------------------------------
        
        w = self._w[min(self._n_generations, len(self._w) - 1)]
        r_mat = self.rng_optimization.random((N, D))
        v = w * v + self.c * r_mat * (n_x - x)
        
        v = np.clip(v, self._min_v, self._max_v)
        x += v
        
        if self.is_bound:
            x = np.clip(x, self.lower_boundary, self.upper_boundary)
            
        # -----------------------------------------------------------------
        # 3. ⭐️ 完全批量化适应度评估 (Batch Fitness Evaluation) ⭐️
        # -----------------------------------------------------------------
        
        # 假设 self._evaluate_fitness 接受 (N, D) 的 x 矩阵并返回 (N,) 的 y 向量
        current_fitness = self._evaluate_fitness(x, args)
        y[:] = current_fitness
        
        # -----------------------------------------------------------------
        # 4. 批量化个人最优位置 (Pbest) 和标志 (Flag) 更新
        # -----------------------------------------------------------------
        
        improved_mask = y < p_y
        
        if np.any(improved_mask):
            p_x[improved_mask] = x[improved_mask]
            p_y[improved_mask] = y[improved_mask]
            self._flag[improved_mask] = 0
            
        self._flag[~improved_mask] += 1
            
        self._n_generations += 1
        
        return v, x, y, p_x, p_y, n_x