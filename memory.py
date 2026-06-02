import numpy as np
import random


class SumTree:
    """基于线段树 (SumTree) 的高效优先级数据结构
    
    add / update / get_leaf 时间复杂度均为 O(log N)
    循环写入指针在容量满后自动覆盖最旧的记录
    """

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity 必须大于 0")
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_ptr = 0

    def add(self, priority, data):
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)

        self.write_ptr += 1
        if self.write_ptr >= self.capacity:
            self.write_ptr = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        priority = self._sanitize_priority(priority)
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx  = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

    @staticmethod
    def _sanitize_priority(priority):
        priority = float(np.nan_to_num(priority, nan=0.0, posinf=0.0, neginf=0.0))
        return max(priority, 0.0)


class PrioritizedReplayBuffer:
    """TD-Error 优先经验回放池 (Prioritized Experience Replay)
    
    alpha  : 优先级指数，0=均匀采样，1=完全按优先级采样，默认 0.6
    beta   : 重要性采样修正指数，训练中从初始值退火至 1.0
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree                      = SumTree(capacity)
        self.alpha                     = alpha
        self.beta                      = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.absolute_error_upper      = 20.0   # TD-Error 上限，防止单条记忆优先级过高
        self.epsilon                   = 0.00001  # 防止优先级为 0 导致永不被采样

    def add(self, experience):
        # 新经验以当前最大优先级加入，确保至少被采样一次
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if not np.isfinite(max_priority) or max_priority <= 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)

    def sample(self, n):
        """分层采样 n 条经验，返回 (batch, tree_idxs, is_weights)"""
        if self.tree.n_entries == 0:
            raise ValueError("经验池为空，无法采样")
        if n <= 0:
            raise ValueError("采样数量必须大于 0")
        n = min(n, self.tree.n_entries)

        batch      = []
        idxs       = []
        priorities = []
        total_priority = self.tree.total_priority
        if not np.isfinite(total_priority) or total_priority <= 0:
            raise ValueError("经验池优先级总和无效，无法采样")

        segment = total_priority / n

        # β 退火：随采样次数增加，IS 权重逐渐趋近均匀（减小偏差修正幅度）
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        priorities = np.maximum(np.asarray(priorities, dtype=np.float64), self.epsilon)
        sampling_probabilities = priorities / total_priority
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        max_weight = is_weights.max()
        if np.isfinite(max_weight) and max_weight > 0:
            is_weights /= max_weight  # 归一化，确保权重不超过 1
        else:
            is_weights = np.ones_like(is_weights, dtype=np.float64)

        return batch, idxs, is_weights.astype(np.float32)

    def update_priorities(self, tree_idxs, td_errors):
        # 修复：训练异常时 td_errors 可能含 NaN 或 Inf
        # 若不过滤，NaN 优先级会向上传播污染整棵 SumTree，导致后续采样永久失效
        td_errors = np.nan_to_num(
            td_errors,
            nan=self.epsilon,                  # NaN -> 最小优先级
            posinf=self.absolute_error_upper,  # +Inf -> 上限值
            neginf=self.epsilon                # -Inf -> 最小优先级
        )
        td_errors += self.epsilon                                        # 保证优先级 > 0
        clipped_errors = np.minimum(td_errors, self.absolute_error_upper)  # 截断上限
        ps = np.power(clipped_errors, self.alpha)
        for idx, p in zip(tree_idxs, ps):
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
