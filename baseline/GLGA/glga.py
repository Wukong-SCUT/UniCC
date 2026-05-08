import numpy as np

from baseline.GLGA.ga import GA


class GLGA(GA):
    """Global and Local genetic algorithm (GL25).

    .. note:: `25` means that 25 percentage of function evaluations (or runtime) are first used for *global* search
       while the remaining 75 percentage are then used for *local* search.

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
                * 'alpha'           - global step-size for crossover (`float`, default: `0.8`),
                * 'n_female_global' - number of female at global search stage (`int`, default: `200`),
                * 'n_male_global'   - number of male at global search stage (`int`, default: `400`),
                * 'n_female_local'  - number of female at local search stage (`int`, default: `5`),
                * 'n_male_local'    - number of male at local search stage (`int`, default: `100`),
                * 'p_global'        - percentage of global search stage (`float`, default: `0.25`),.

    Examples
    --------
    Use the optimizer to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.ga.gl25 import GL25
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5*numpy.ones((2,)),
       ...            'upper_boundary': 5*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022}
       >>> gl25 = GL25(problem, options)  # initialize the optimizer class
       >>> results = gl25.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"GL25: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       GL25: 5000, 1.0505276479694516e-05

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/ytzffmbc>`_ for more details.

    Attributes
    ----------
    alpha           : `float`
                      global step-size for crossover.
    n_female_global : `int`
                      number of female at global search stage.
    n_female_local  : `int`
                      number of female at local search stage.
    n_individuals   : `int`
                      population size.
    n_male_global   : `int`
                      number of male at global search stage.
    n_male_local    : `int`
                      number of male at local search stage.
    p_global        : `float`
                      percentage of global search stage.

    References
    ----------
    García-Martínez, C., Lozano, M., Herrera, F., Molina, D. and Sánchez, A.M., 2008.
    Global and local real-coded genetic algorithms based on parent-centric crossover operators.
    European Journal of Operational Research, 185(3), pp.1088-1113.
    https://www.sciencedirect.com/science/article/abs/pii/S0377221706006308
    """
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.alpha = options.get('alpha', 0.8)
        assert self.alpha > 0.0
        self.p_global = options.get('p_global', 0.25)  # percentage of global search stage
        assert 0.0 <= self.p_global <= 1.0
        self.n_female_global = options.get('n_female_global', 200)  # number of female at global search stage
        assert self.n_female_global > 0
        self.n_male_global = options.get('n_male_global', 400)  # number of male at global search stage
        assert self.n_male_global > 0
        self.n_female_local = options.get('n_female_local', 5)  # number of female at local search stage
        assert self.n_female_local > 0
        self.n_male_local = options.get('n_male_local', 100)  # number of male at local search stage
        assert self.n_male_local > 0
        self.n_individuals = int(np.maximum(self.n_male_global, self.n_male_local))
        self._assortative_mating = 5
        self._n_selected = np.zeros((self.n_individuals,))  # number of individuals selected as female
        # set maximum of function evaluations and runtime for global search stage
        self._max_fe_global = self.p_global*self.max_function_evaluations
        self._max_runtime_global = self.p_global*self.max_runtime

    def initialize(self, args=None):
        x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                             size=(self.n_individuals, self.ndim_problem))  # population
        
        # --- [修改开始] ---
        if self._check_terminations():
            y = np.empty((self.n_individuals,))
        else:
            # 批量评估整个初始种群
            y = self._evaluate_fitness(x, args)
        # --- [修改结束] ---
            
        return x, y

    def iterate(self, x=None, y=None, n_female=None, n_male=None, args=None):
        # n_male 将作为本世代生成的子代数量 (Batch Size)
        batch_size = n_male 
        
        # 1. 选择亲代
        order = np.argsort(y)
        x_all_male = x[order[range(n_male)]]
        
        # 针对每个子代，都需要独立选择一对亲代。这里将选择过程向量化/批量化。
        
        # 批量选择 female (使用 UFS)
        # 选择 n_female 个最佳个体作为潜在 female
        potential_female_indices = order[range(n_female)]
        _n_selected = self._n_selected[potential_female_indices]
        
        # 为 batch_size 个子代生成 batch_size 对亲代 (female 和 male)
        
        # *a*. 批量选择 Female (使用 UFS 的简化批量实现)
        # 在这里，我们简化为：从潜在 female 中随机选择 batch_size 个 female 亲代
        # 实际 GL25 的 UFS 旨在保证公平选择，这里使用随机替换模拟批量选择。
        # 亲代选择是在循环之前完成的，但我们现在需要 batch_size 个 female。
        female_indices_batch = self.rng_optimization.choice(
            potential_female_indices, size=batch_size, replace=True)
        x_female_batch = x[female_indices_batch]
        
        # 更新 _n_selected (这里只记录了被选中的次数，不影响并行性)
        for idx in female_indices_batch:
            self._n_selected[idx] += 1
        
        # *b*. 批量选择 Male (使用 NAM 的简化批量实现)
        # 从 x_all_male (n_male 个最佳个体) 中为每个 female 批量选择一个最远的 male
        # 注意：严格的 NAM 需要计算距离矩阵，这里为了批量化和简化，我们批量选择 batch_size 个 male。
        
        # 批量选择 _assortative_mating 个候选中最远的一个作为 male
        x_male_batch = np.empty((batch_size, self.ndim_problem))
        for k in range(batch_size):
            # 随机选择 _assortative_mating 个候选 male
            male_candidates_indices = self.rng_optimization.choice(n_male, size=self._assortative_mating, replace=False)
            male_candidates = x_all_male[male_candidates_indices]
            
            # 计算距离并选择最远的一个 (NAM 策略)
            distances = np.linalg.norm(x_female_batch[k] - male_candidates, axis=1)
            x_male_batch[k] = male_candidates[np.argmax(distances)]
        
        # 2. 批量生成子代 (PC-BLX 交叉)
        # female, male, xx 都是 size=(batch_size, ndim_problem) 的矩阵
        interval = np.abs(x_female_batch - x_male_batch)
        l, u = x_female_batch - interval * self.alpha, x_female_batch + interval * self.alpha
        
        # 确保上下界在搜索范围内
        lower_bound_matrix = np.tile(self.lower_boundary, (batch_size, 1))
        upper_bound_matrix = np.tile(self.upper_boundary, (batch_size, 1))

        # 批量生成子代 (xx)
        xx = self.rng_optimization.uniform(
            np.clip(l, lower_bound_matrix, upper_bound_matrix),
            np.clip(u, lower_bound_matrix, upper_bound_matrix),
            size=(batch_size, self.ndim_problem)
        )
        
        # 3. 批量评估子代 (并行评估点)
        yy = self._evaluate_fitness(xx, args) 
        
        # 4. 批量替换最差策略 (RW)
        # 获取种群中 y 最差的 batch_size 个个体的索引 (它们将被替换)
        order = np.argsort(y)
        worst_indices = order[-batch_size:]
        
        # 找出哪些新子代 yy 优于待替换的旧个体 y[worst_indices]
        # 注意：这里需要一个更复杂的逻辑来匹配新子代和旧个体，
        # 为了最小化修改，我们简化为：如果新子代的最佳优于旧个体的最差，则替换。
        # 
        # 假设：简单地用这 batch_size 个子代替换种群中最差的 batch_size 个个体
        
        # 找出新子代中优于旧个体最差值 (y[order[-1]]) 的个体
        
        # 找出所有比种群中最差个体 y[order[-1]] 更好的子代
        better_mask = yy < y[order[-1]]
        
        if np.any(better_mask):
            # 将这些更好的子代 xx[better_mask] 放入种群中
            
            # 找出需要被替换的种群中最差个体 (数量等于 better_mask 中 True 的个数)
            num_replacements = np.sum(better_mask)
            indices_to_replace = order[-num_replacements:]
            
            # 执行替换 (RW 策略)
            x[indices_to_replace] = xx[better_mask]
            y[indices_to_replace] = yy[better_mask]
            
            # 重置被替换个体的 _n_selected 计数
            self._n_selected[indices_to_replace] = 0
            
            # 返回最好的适应度值（用于打印信息）
            yy_out = np.min(yy[better_mask])
        else:
            yy_out = np.min(yy)
            
        self._n_generations += 1
        return x, yy_out

    def optimize(self, fitness_function=None, args=None):
        fitness, is_switch = GA.optimize(self, fitness_function), True
        x, y = self.initialize(args)
        yy = y  # only for printing
        while not self._check_terminations():
            self._print_verbose_info(fitness, yy)
            if self.n_function_evaluations >= self._max_fe_global or self.runtime >= self._max_runtime_global:
                if is_switch:  # local search
                    init, is_switch = range(np.maximum(self.n_female_local, self.n_male_local)), False
                    x, y, self._n_selected = x[init], y[init], self._n_selected[init]
                x, yy = self.iterate(x, y, self.n_female_local, self.n_male_local, args)
            else:  # global search
                x, yy = self.iterate(x, y, self.n_female_global, self.n_male_global, args)
        return self._collect(fitness, yy)