# 🐍 Snake AI — Double DQN + PER

基于 **Double DQN**、**优先经验回放（PER）**、**混合精度训练（AMP）** 的贪吃蛇强化学习项目。  
使用 PyTorch + Pygame 构建，支持 TensorBoard 实时监控与断点续训。

---

## 目录

- [效果预览](#效果预览)
- [技术架构](#技术架构)
- [算法原理](#算法原理)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [状态空间设计](#状态空间设计)
- [奖励函数设计](#奖励函数设计)
- [训练监控](#训练监控)
- [文件详解](#文件详解)
- [已知局限与未来方向](#已知局限与未来方向)

---

## 效果预览

训练过程中，智能体从完全随机探索逐步学会：

- 避免撞墙和撞到自身
- 规划路径向食物靠近
- 在蛇体较长时绕行而不是直接冲向食物

训练约 **1000 局**后开始出现稳定的食物获取行为，**5000 局**后探索率降至 1% 并进入纯利用阶段。

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        agent.py                             │
│  Agent                                                      │
│  ├── get_state()      将游戏画面编码为 4×24×32 张量         │
│  ├── get_action()     ε-greedy 探索策略                     │
│  ├── remember()       存入 PER 经验池                       │
│  ├── train_short_memory()  单步在线训练                     │
│  └── train_long_memory()   批量 PER 采样训练               │
└────────────┬──────────────────┬──────────────────┬──────────┘
             │                  │                  │
      ┌──────▼──────┐   ┌───────▼──────┐   ┌──────▼──────┐
      │   game.py   │   │   model.py   │   │  memory.py  │
      │             │   │              │   │             │
      │SnakeGameAI  │   │ Conv_QNet    │   │ SumTree     │
      │ ├─ reset()  │   │ ├─ conv1     │   │ PrioritizedR│
      │ ├─ play_    │   │ ├─ conv2     │   │ eplayBuffer │
      │ │  step()   │   │ ├─ fc1       │   │             │
      │ └─ is_      │   │ └─ fc2       │   │ O(log N)    │
      │  collision()│   │              │   │ 采样        │
      │             │   │ QTrainer     │   │             │
      └─────────────┘   │ └─ Double   │   └─────────────┘
                        │   DQN+AMP   │
                        └─────────────┘
```

---

## 算法原理

### Double DQN

标准 DQN 使用同一个网络同时选择动作和估计 Q 值，容易产生过估计偏差。Double DQN 将两个步骤分离：

```
# Online 网络：选择最优动作
best_action = argmax(Q_online(s'))

# Target 网络：估计该动作的 Q 值
Q_target = r + γ · Q_target(s', best_action)
```

Target 网络通过 **Soft Update** 缓慢跟随 Online 网络（τ=0.005），避免训练目标剧烈震荡：

```
θ_target ← τ·θ_online + (1 - τ)·θ_target
```

### 优先经验回放（PER）

基于 SumTree 数据结构实现 O(log N) 的优先级采样。TD-Error 越大的经验（网络犯错越多的情况），被采样的概率越高：

```
P(i) = p_i^α / Σ p_k^α
```

为纠正优先采样引入的分布偏差，使用重要性采样权重（IS Weights）修正梯度：

```
Loss = mean( IS_weights · (Q_target - Q_pred)² )
```

β 参数从 0.4 线性退火至 1.0，随训练进行逐渐消除偏差修正。

### 奖励塑形

| 事件 | 奖励值 | 说明 |
|------|--------|------|
| 吃到食物 | +10 | 核心驱动信号 |
| 死亡（撞墙/撞身） | -10 | 死亡惩罚 |
| 靠近食物（曼哈顿距离减小） | +0.02 | 方向引导辅助信号 |
| 远离食物（曼哈顿距离增大） | -0.02 | 远离惩罚 |
| 距离不变 | 0 | 中性 |

辅助信号幅度极小（最大累计 ±1.12），绝对不会覆盖 ±10 的核心奖励，仅提供方向引导。

---

## 项目结构

```
snake-ai/
│
├── game.py          # 游戏环境：SnakeGameAI，负责物理逻辑与渲染
├── model.py         # 神经网络：Conv_QNet + QTrainer（Double DQN）
├── agent.py         # 智能体：状态编码、策略、训练循环
├── memory.py        # 经验回放：SumTree + PrioritizedReplayBuffer
│
├── model/           # 自动生成，存放训练存档
│   ├── checkpoint.pth   # 定期存档（每 50 局）
│   └── best_model.pth   # 历史最高分存档
│
├── logs/            # TensorBoard 日志（自动生成）
│   └── Snake_CNN_PER_<timestamp>/
│
└── README.md
```

---

## 环境要求

**Python 版本：** 3.9+

**依赖库：**

```
torch >= 2.0.0
pygame >= 2.0.0
numpy >= 1.21.0
tensorboard >= 2.10.0
```

**安装依赖：**

```bash
pip install torch pygame numpy tensorboard
```

**GPU 支持（可选）：**

代码会自动检测 CUDA 是否可用并启用混合精度训练。无 GPU 时自动降级为 CPU 运行，所有代码路径均兼容。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd snake-ai
```

### 2. 安装依赖

```bash
pip install torch pygame numpy tensorboard
```

### 3. 开始训练

```bash
python agent.py
```

首次运行会自动创建 `model/` 目录。程序启动后：

- 初始 **5000 局**为探索阶段，ε 从 100% 线性衰减至 1%
- 每 **50 局**自动保存一次 `checkpoint.pth`
- 破纪录时自动保存 `best_model.pth`
- 每 50 局或当前分数超过 30 时开启画面渲染，其余时间全速训练

### 4. 继续上次训练

直接再次运行即可，程序会自动加载 `checkpoint.pth`：

```bash
python agent.py
# 输出：🎉 成功恢复训练现场！当前局数: 1200 | 历史最高分: 18
```

### 5. 实时监控训练

在另一个终端运行：

```bash
tensorboard --logdir=logs
```

打开浏览器访问 `http://localhost:6006`，可以实时查看：

- `Loss/Training`：批量训练损失
- `Score/Current`：每局得分
- `Score/Record`：历史最高分
- `Metrics/Epsilon`：当前探索率

---

## 配置说明

主要超参数位于 `agent.py` 顶部，可直接修改：

```python
MAX_MEMORY = 100000   # PER 经验池容量
BATCH_SIZE = 1024     # 每次批量训练的样本数
LR         = 0.0003   # Adam 优化器学习率
```

网络结构超参数位于 `model.py`：

```python
# Conv_QNet 默认参数（与游戏分辨率对应）
Conv_QNet(in_channels=4, grid_h=24, grid_w=32, output_size=3)
```

PER 超参数位于 `memory.py`：

```python
PrioritizedReplayBuffer(
    capacity=MAX_MEMORY,
    alpha=0.6,                       # 优先级指数（0=均匀, 1=完全优先）
    beta=0.4,                        # IS 权重初始值（退火至 1.0）
    beta_increment_per_sampling=0.001 # β 每次采样增量
)
```

探索策略在 `agent.py` 的 `get_action()` 中：

```python
self.epsilon = max(1, 100 - (self.n_games / 50))
# 前 5000 局：100% → 1% 线性衰减
# 5000 局后：锁定在 1% 保持少量随机探索
```

---

## 状态空间设计

智能体将游戏画面编码为一个 **4 通道 24×32 的张量**，送入 CNN 处理：

```
状态张量 shape: (4, 24, 32)

通道 0：食物位置     稀疏二值图，食物所在格子 = 1.0，其余 = 0.0
通道 1：蛇头位置     稀疏二值图，蛇头所在格子 = 1.0，其余 = 0.0
通道 2：蛇身位置     稀疏二值图，蛇身格子 = 1.0，其余 = 0.0
通道 3：当前方向     全图填充同一浮点值（UP=0.25, RIGHT=0.5, DOWN=0.75, LEFT=1.0）
```

游戏窗口 640×480 像素，网格大小 20×20，因此网格为 32 列 × 24 行。

---

## 奖励函数设计

完整的奖励逻辑（位于 `game.py` 的 `play_step()`）：

```python
# 优先判断：死亡（最高优先级，直接返回）
if is_collision() or frame_iteration > 100 * len(snake):
    return reward=-10, game_over=True

# 次要判断：吃到食物
if head == food:
    score += 1
    return reward=+10

# 默认：距离塑形
else:
    snake.pop()  # 尾部收缩
    if new_distance < old_distance:
        reward = +0.02
    elif new_distance > old_distance:
        reward = -0.02
    else:
        reward = 0
```

超时机制防止绕圈：允许步数 = `100 × 蛇的当前长度`，随着蛇变长自动增加探索时间。

---

## 训练监控

### TensorBoard 指标说明

| 指标名 | 含义 | 预期趋势 |
|--------|------|---------|
| `Loss/Training` | 每局结束后批量训练的 MSE Loss | 早期震荡，后期应逐渐下降并趋于平稳 |
| `Score/Current` | 每局游戏得分 | 随训练局数增加，均值应缓慢上升 |
| `Score/Record` | 历史最高分 | 单调不降的阶梯型曲线 |
| `Metrics/Epsilon` | 当前探索率（%） | 从 100 线性降至 1，之后保持不变 |

### 控制台输出格式

```
Game   150 | Epsilon:  97.0% | Score:   2 | Record:   8 | Loss: 0.012345
Game   200 | Epsilon:  96.0% | Score:   5 | Record:   8 | Loss: 0.009821
```

### 手动停止

按 `Ctrl+C` 可随时安全停止训练，程序会自动保存当前状态后退出：

```
🛑 接收到手动停止信号 (Ctrl+C)，正在保存最后的训练现场...
✅ 存档成功！你可以安全关闭了，下次运行会自动接上进度。
```

---

## 文件详解

### `game.py` — 游戏环境

基于 Pygame 的贪吃蛇环境，实现了标准的 RL 环境接口。

**核心接口：**

```python
game = SnakeGameAI(w=640, h=480)
game.reset()                              # 重置游戏状态
reward, game_over, score = game.play_step(action, render=True)
# action: [1,0,0]=直走 | [0,1,0]=右转 | [0,0,1]=左转（相对当前方向）
```

**关键设计：**
- `render=False` 时跳过所有绘制和帧率限制，训练速度提升 10~50 倍
- `_place_food()` 使用迭代而非递归，避免蛇体极长时的栈溢出

### `model.py` — 神经网络

**Conv_QNet 结构：**

```
输入：(Batch, 4, 24, 32)
  ↓ Conv2d(4→16, kernel=4, stride=2, padding=1) + ReLU
  ↓ Conv2d(16→32, kernel=4, stride=2, padding=1) + ReLU
  ↓ Flatten → (Batch, 1536)
  ↓ Linear(1536 → 256) + ReLU
  ↓ Linear(256 → 3)
输出：(Batch, 3)  ← 三个动作的 Q 值
```

fc1 的输入尺寸 1536（32×6×8）通过 dummy tensor 动态推断，修改卷积参数时自动适配。

**QTrainer：**
- Double DQN 目标值计算
- 带 IS 权重的加权 MSE Loss
- Soft Target Update（τ=0.005）
- 混合精度训练（CUDA 下自动启用 AMP）

### `agent.py` — 智能体

**训练流程（每一步）：**

```
1. get_state(game)         → 获取当前 4×24×32 状态张量
2. get_action(state)       → ε-greedy 选择动作
3. game.play_step(action)  → 执行动作，获得 reward, done, score
4. get_state(game)         → 获取新状态
5. train_short_memory()    → 单步在线训练
6. remember()              → 存入 PER 经验池
7. if done: train_long_memory()  → 批量 PER 采样训练
```

### `memory.py` — 经验回放

**SumTree 结构：**

```
容量为 N 的 SumTree 共有 2N-1 个节点：
  - 叶节点（N个）：存储每条经验的优先级
  - 内部节点（N-1个）：子节点优先级之和

操作复杂度：
  - add()：O(log N)
  - update()：O(log N)
  - get_leaf()：O(log N)  ← 核心优势，标准数组采样为 O(N)
```

---

## 已知局限与未来方向

### 当前局限

**状态表示：** 方向信息用连续标量（0.25/0.5/0.75/1.0）表示类别变量，存在隐式大小关系。更好的做法是用 4 个独立的 one-hot 通道。

**网格尺寸耦合：** `Conv_QNet` 的默认 `grid_h=24, grid_w=32` 与游戏窗口尺寸隐式绑定，修改窗口尺寸时需要同步修改网络初始化参数。

**双训练架构：** 每步同时执行单步在线训练和批量 PER 训练，两者梯度方向可能在探索期存在冲突，建议通过实验验证是否有必要保留单步训练。

### 可选的进阶方向

**Dueling DQN：** 将 Q 值分解为状态价值 V(s) 和动作优势 A(s,a)，通常能提升收敛速度和最终分数上限，代码改动量较小。

**方向通道 one-hot：** 将通道 3 拆分为 4 个独立通道（in_channels 从 4 改为 7），消除类别变量的隐式数值大小关系。

**障碍物支持：** 代码中已有障碍物逻辑的注释框架，取消注释并同步修改状态表示（增加通道 4 表示障碍物）即可启用。

**Noisy Networks：** 用参数化噪声层替代 ε-greedy 探索，让探索更自适应。

---

## 许可证

MIT License — 自由使用、修改与分发。
