# Snake AI

基于 PyTorch 和 Pygame 的贪吃蛇强化学习实验项目。当前版本使用 Double DQN、软更新 target network、优先经验回放（PER）和双模态状态输入训练智能体。

## 项目结构

```text
agent.py      训练入口、状态编码、epsilon-greedy 策略、checkpoint 管理
game.py       贪吃蛇环境、动作转换、奖励函数和可选渲染
model.py      双模态 Q 网络与 Double DQN 训练步骤
memory.py     SumTree 和 PrioritizedReplayBuffer
test_core.py  核心链路单元测试
```

## 状态空间

`Agent.get_state()` 返回 `(image_state, vector_features)`：

```text
image_state shape: (3, 24, 32)
  channel 0: 食物位置
  channel 1: 蛇头位置
  channel 2: 蛇身位置，越靠近头部数值越高

vector_features shape: (11,)
  [danger_straight, danger_right, danger_left,
   dir_left, dir_right, dir_up, dir_down,
   food_left, food_right, food_up, food_down]
```

这种设计让 CNN 保留棋盘空间结构，同时把 DQN 最需要的局部危险、当前方向和食物相对位置直接提供给 MLP 分支，避免模型从稀疏图像里低效推断这些基础关系。

## 算法与训练

- **Double DQN**：online 网络选择下一步动作，target 网络估计该动作价值，降低 Q 值过估计风险。
- **Soft target update**：每次训练后用 `tau=0.005` 平滑更新 target network。
- **PER**：用 SumTree 根据 TD error 采样经验，并使用 importance-sampling weights 修正偏差。
- **Huber loss**：替代普通 MSE，降低异常 TD error 对训练的冲击。
- **AMP 与梯度裁剪**：CUDA 环境自动启用 AMP；每步训练进行梯度裁剪。

## 奖励函数

| 事件 | 奖励 |
| --- | --- |
| 吃到食物 | `+10` |
| 撞墙、撞自己或超时 | `-10` |
| 曼哈顿距离靠近食物 | `+0.02` |
| 曼哈顿距离远离食物 | `-0.02` |
| 距离不变 | `0` |

## 快速开始

```bash
pip install -r requirements.txt
python agent.py
```

训练会自动在 `model/` 下保存 checkpoint，在 `logs/` 下写入 TensorBoard 日志。查看曲线：

```bash
tensorboard --logdir=logs
```

## 测试

```bash
python -m unittest -v test_core.py
```

测试覆盖：

- 状态编码形状与方向 one-hot
- 动作选择、经验写入和单步训练
- PER 采样与异常优先级清洗
- batch 训练链路
- checkpoint 保存与恢复
- 食物生成边界条件

## 主要超参数

核心配置集中在 `agent.py` 的 `TrainConfig`：

```python
max_memory = 100_000
batch_size = 256
warmup_steps = 2_000
train_every_steps = 4
learning_rate = 0.0003
gamma = 0.99
epsilon_decay_games = 5_000
```

## 当前定位

这是一个可运行、可测试的强化学习实验项目，不是已完成性能基准的生产系统。模型效果仍需要通过长时间训练、多随机种子实验和评估脚本进一步验证。

## 许可证

本项目采用 GPLv3 许可证，详见 `LICENSE`。
