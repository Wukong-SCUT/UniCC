import numpy as np
import matplotlib.pyplot as plt

class fun_record:
    def __init__(self, fun, best, dims, CC=True):
        self.fun = fun  # objective function
        self.best = best  # the best solution
        self.dims = dims  # the dimension of the subgroup
        self.CC = CC  # whether the action is CC
        self.fitness_record = []  # record the fitness of the offspring of each iteration
        self.individual_record = [] # record all of the individuals in one step
    
    def combine(self, small_vec, background_vec, location):
        if location is None:
            return small_vec
        else:
            combination = np.tile(background_vec, (len(small_vec), 1))
            combination[:, location] = small_vec
            return combination
    
    def __call__(self, X_batch):
        if self.CC:
            X_transform = self.combine(X_batch, self.best, self.dims)
            self.individual_record.extend(X_transform.copy())
            OBJFUNC = lambda X_batch: self.fun(X_transform)
            fitness = OBJFUNC(X_batch)
            self.fitness_record.extend(fitness.copy())
        else:
            self.individual_record.extend(X_batch.copy())
            fitness = self.fun(X_batch)
            self.fitness_record.extend(fitness.copy())
        return fitness

def partition_p_and_s(p_file_path, s_file_path, overlap=0):
    # 读取文件内容
    def read_file(file_path):
        with open(file_path, "r") as file:
            content = file.read().strip()
        return content

    # 解析数据为列表
    def parse_p_values(p_content):
        return list(map(int, p_content.split(",")))

    def parse_s_values(s_content):
        return list(map(int, s_content.split()))

    # 划分维度数据，支持重叠
    def partition_p_by_s(p_values, s_values, overlap):
        partitioned = []
        start_index = 0

        for size in s_values:
            # 如果不是第一个分区，应用重叠
            if partitioned and overlap > 0:
                start_index = start_index - overlap
            end_index = start_index + size
            partitioned.append(p_values[start_index:end_index])
            start_index = end_index

        return partitioned

    # 读取文件内容
    p_content = read_file(p_file_path)
    s_content = read_file(s_file_path)

    # 解析文件数据
    p_values = parse_p_values(p_content)
    p_values =[x - 1 for x in p_values]
    s_values = parse_s_values(s_content)

    # 根据 s 划分 p，支持重叠
    return partition_p_by_s(p_values, s_values, overlap)

def make_monotonic_decreasing(arr):
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:  # 如果前面的元素小于后面的元素
            arr[i + 1] = arr[i]  # 修改后面的元素，使其不大于当前元素
    return arr

def evaluation_record(data, output_path, record_FEs_list):
    """
    记录算法的评估值（包括特定评估点和最终点），以及运行时间。

    Args:
        data (dict): 包含算法运行结果及运行时间的字典。
        output_path (str): 输出路径。
        record_FEs_list (list): 特定的评估点列表。
    """
    # Convert the record points to integers
    record_FEs_list = [int(x) for x in record_FEs_list]
    
    # Initialize a dictionary to hold the average fitness values for each algorithm
    algorithm_avg_fitness = {}

    for algorithm, runs in data.items():
        if "_time" in algorithm:  # 跳过时间记录的键
            continue
        runs = [make_monotonic_decreasing(run.copy()) for run in runs]
        # 获取当前算法的所有运行数据
        max_length = max(len(run) for run in runs)  # 获取所有运行中最长的评估值列表长度
        avg_fitness = []  # 存储每个评估次数的均值
        variances = []  # 存储每个评估次数的方差

        # 遍历每个评估次数
        for i in range(max_length):
            values_at_i = [run[i] for run in runs if i < len(run)]  # 获取所有运行中第 i 次评估的值
            if values_at_i:  # 如果有运行达到了当前评估次数
                avg_fitness.append(sum(values_at_i) / len(values_at_i))  # 计算均值
                variances.append(np.std(values_at_i)) # 计算方差
            else:
                avg_fitness.append(None)  # 如果没有值，填充 None
                variances.append(None)  # 方差也填充 None
        
        # Store the computed average fitness for the current algorithm
        algorithm_avg_fitness[algorithm] = {'avg_fitness': avg_fitness, 'variances': variances}
    
    # Save the record points and their corresponding fitness values to a txt file
    output_file_path = f"{output_path}evaluation_record.txt"
    with open(output_file_path, 'w') as f:
        # Write a well-formatted header with a line separator
        f.write(f"{'Algorithm':<20}{'Record Point':<25}{'Fitness Value':<30}{'Scientific Notation':<25}{'Variance':<30}{'Scientific Notation':<25}\n")
        f.write("-" * 155 + "\n")  # Separator line
        
        # Write each record point for every algorithm
        for algorithm, avg_fitness in algorithm_avg_fitness.items():
            f.write(f"Algorithm: {algorithm}\n")  # Write the algorithm name
            
            # 获取该算法的运行时间
            time_key = f"{algorithm}_time"
            if time_key in data:
                # 如果存在时间记录，获取运行时间均值和标准差
                run_time_avg = np.mean(data[time_key])
                run_time_std = np.std(data[time_key])
            else:
                run_time_avg = None  # 如果没有时间记录，设为 None
                run_time_std = None
            
            for record_FEs in record_FEs_list:
                if record_FEs < len(algorithm_avg_fitness[algorithm]['avg_fitness']):
                    fitness_value = algorithm_avg_fitness[algorithm]['avg_fitness'][record_FEs]
                    variance_value = algorithm_avg_fitness[algorithm]['variances'][record_FEs]
                    if fitness_value is not None:
                        # Format both decimal and scientific notation
                        f.write(f"{'':<20}{record_FEs:<25.3e}{fitness_value:<30.6f}{fitness_value:<25.6e}{variance_value:<30.6f}{variance_value:<25.6e}\n")
                else:
                    print(f"Warning: Record point {record_FEs} exceeds the available evaluations for {algorithm}.")
            
            # 记录最终点
            final_value = algorithm_avg_fitness[algorithm]['avg_fitness'][-1] if algorithm_avg_fitness[algorithm]['avg_fitness'][-1] is not None else "N/A"
            final_variance = algorithm_avg_fitness[algorithm]['variances'][-1] if algorithm_avg_fitness[algorithm]['variances'][-1] is not None else "N/A"
            inner_value = f"Fin:{len(algorithm_avg_fitness[algorithm]['avg_fitness']):.3e}"
            f.write(f"{'':<20}{inner_value:<25}{final_value:<30.6f}{final_value:<25.6e}{final_variance:<30.6f}{final_variance:<25.6e}\n")
            
            # 添加运行时间记录
            if run_time_avg is not None:
                f.write(f"{'':<20}{'Run Time:':<25}{run_time_avg:<30.6f}\n")
                f.write(f"{'':<20}{'Run Time Std:':<25}{run_time_std:<30.6f}\n")
                
            # Add a separator between algorithms
            f.write("-" * 155 + "\n")
    
    print(f"Evaluation records have been saved to '{output_file_path}'.")

def plot_evaluation_curve_best_so_far(
    data, output_path, font_size, maxfes, log_scale=False, show_variance=False, eps=1e-12
):
    """
    中心曲线：算术均值（linear）
    方差带：在 log10 空间计算标准差，形成乘法对称带（/factor 与 *factor）
    """
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    for algorithm, runs in data.items():
        if "_time" in algorithm:
            continue

        # 每条 run 先做 best-so-far（单调不增）
        runs = [make_monotonic_decreasing(run.copy()) for run in runs]

        # 以最长 run 为准逐点统计
        max_length = min(int(maxfes),min(len(run) for run in runs))  # 获取所有运行中最长的评估值列表长度

        xs, centers, lows, highs = [], [], [], []
        for i in range(max_length):
            values_at_i = [run[i] for run in runs if i < len(run)]
            if not values_at_i:
                continue

            vals = np.asarray(values_at_i, dtype=float)
            vals = np.clip(vals, eps, None)  # 防止非正导致 log 失败

            # 中心：算术均值（不要再对聚合后的均值做单调化）
            c = vals.mean()
            xs.append(i)
            centers.append(c)

            # 方差带：log 空间 std -> 乘法因子
            if show_variance:
                std_log = np.log10(vals).std()
                factor = 10 ** std_log
                lows.append(c / factor)
                highs.append(c * factor)

        # 画均值
        line, = ax.plot(xs, centers, label=algorithm)

        # 画方差带
        if show_variance and len(xs) > 0:
            ax.fill_between(xs, lows, highs, color=line.get_color(), alpha=0.2)

    if log_scale:
        ax.set_yscale("log")

    plt.rcParams.update({'font.size': font_size})
    ax.set_xlabel("FEs", fontsize=font_size)
    ax.set_ylabel("Objective Value (log10)", fontsize=font_size)
    ax.set_title("Best-so-Far Evaluation Curves for Different Algorithms", fontsize=font_size)
    ax.legend(fontsize=font_size)
    ax.grid(True)

    filename = "evaluation_curves_best_so_far.png"
    plt.savefig(f"{output_path}{filename}", bbox_inches='tight')
    print(f"Plot saved to '{output_path}{filename}'.")
    plt.close()