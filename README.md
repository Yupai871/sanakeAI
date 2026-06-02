# Snake AI# 贪吃蛇AI项目
基于GD32单片机的贪吃蛇游戏AI实现

## 本地运行记录
- 2026-06-02：本地成功编译并运行项目
- 开发环境：Keil MDK 5
- 硬件平台：GD32F103C8T6

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

## 使用训练好的模型

训练完成后会生成：

```text
model/best_model.pth      历史最高分模型
model/checkpoint.pth      最近一次周期性保存
```

如果使用本次训练出的新模型，目录是：

```text
model_v2/best_model.pth
model_v2/checkpoint.pth
```

在 Pygame 可视化窗口里运行最佳模型：

```bash
python agent.py --play --model-dir ./model_v2 --model-file best_model.pth
```

常用参数：

```bash
# 调整显示速度
python agent.py --play --model-dir ./model_v2 --fps 20

# 使用最近 checkpoint 而不是 best model
python agent.py --play --model-dir ./model_v2 --model-file checkpoint.pth
```

运行后会打开贪吃蛇窗口，模型会自动控制蛇行动；关闭窗口即可退出。

## 学生 Git 练习流程

### 1. 拉取项目

```bash
git clone https://gitee.com/yupai871---harmony-drumbeat/snake-ai.git
cd snake-ai
```

如果老师要求使用 GitHub，请把地址换成对应 GitHub 仓库地址。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 创建自己的分支

分支名建议使用 `student/学号-姓名`：

```bash
git checkout -b student/20260001-zhangsan
```

### 4. 运行训练好的模型

```bash
python agent.py --play --model-dir ./model_v2 --model-file best_model.pth --fps 30
```

每局结束后终端会输出：

```text
Run 1 | Score 24 | Steps 520
Run 2 | Score 31 | Steps 640
```

至少记录 5 局结果。

### 5. 填写自己的结果文件

复制模板：

```bash
copy student_results\template.csv student_results\20260001_张三.csv
```

把里面的示例数据改成自己的运行结果。

要求：

- 每位同学只修改自己的 `student_results/学号_姓名.csv`
- 不要修改其他同学的文件
- 不要提交 `model/`、`model_v2/`、`logs/`、`artifacts/` 里的训练输出

### 6. 提交并推送

```bash
git status
git add student_results/20260001_张三.csv
git commit -m "submit: 20260001 张三 snake ai result"
git push origin student/20260001-zhangsan
```

### 7. 提交给老师检查

把你的分支名发给老师，例如：

```text
student/20260001-zhangsan
```

老师可以通过该分支检查你的 Git 操作和运行结果。

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
