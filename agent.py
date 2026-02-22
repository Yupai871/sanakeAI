import torch
import random
import numpy as np
import os
import time
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Conv_QNet, QTrainer
from torch.utils.tensorboard import SummaryWriter
from memory import PrioritizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_MEMORY = 100000
BATCH_SIZE = 256
LR         = 0.0003
MODEL_DIR  = './model'


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma   = 0.9

        self.memory = PrioritizedReplayBuffer(capacity=MAX_MEMORY)

        self.model = Conv_QNet()
        self.model.to(device)

        self.target_model = Conv_QNet()
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

    # ── Checkpoint ──────────────────────────────────────────────────────────
    def save_checkpoint(self, record, filename='checkpoint.pth'):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, filename)
        checkpoint = {
            'n_games':            self.n_games,
            'record':             record,
            'model_state':        self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),
            'optimizer_state':    self.trainer.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename='checkpoint.pth'):
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            # 修复：补齐 weights_only=True，消除 PyTorch 2.x FutureWarning
            checkpoint = torch.load(path, weights_only=True)
            self.n_games = checkpoint['n_games']
            self.model.load_state_dict(checkpoint['model_state'])
            self.target_model.load_state_dict(checkpoint['target_model_state'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"🎉 成功恢复训练现场！当前局数: {self.n_games} | 历史最高分: {checkpoint['record']}")
            return checkpoint['record']
        else:
            print("👶 没有找到存档，这是一个全新的开始！")
            return 0

    # ── State ────────────────────────────────────────────────────────────────

    def get_state(self, game):
        """
        将游戏状态编码为 7 通道的 24×32 张量：
          通道 0：食物位置
          通道 1：蛇头位置
          通道 2：蛇身位置
          通道 3：方向 向上 (UP)
          通道 4：方向 向右 (RIGHT)
          通道 5：方向 向下 (DOWN)
          通道 6：方向 向左 (LEFT)
        """
        # 【修改1】把 4 通道改为 7 通道
        state = np.zeros((7, 24, 32), dtype=np.float32)

        # 通道 0：食物
        fx, fy = int(game.food.x // BLOCK_SIZE), int(game.food.y // BLOCK_SIZE)
        if 0 <= fx < 32 and 0 <= fy < 24:
            state[0, fy, fx] = 1.0

        # 通道 1：蛇头
        hx, hy = int(game.head.x // BLOCK_SIZE), int(game.head.y // BLOCK_SIZE)
        if 0 <= hx < 32 and 0 <= hy < 24:
            state[1, hy, hx] = 1.0

        # 通道 2：蛇身（不含头部）
        for pt in game.snake[1:]:
            bx, by = int(pt.x // BLOCK_SIZE), int(pt.y // BLOCK_SIZE)
            if 0 <= bx < 32 and 0 <= by < 24:
                state[2, by, bx] = 1.0

        # 【修改2】方向的 One-hot 编码铺满对应的专属通道
        if game.direction == Direction.UP:
            state[3, :, :] = 1.0
        elif game.direction == Direction.RIGHT:
            state[4, :, :] = 1.0
        elif game.direction == Direction.DOWN:
            state[5, :, :] = 1.0
        elif game.direction == Direction.LEFT:
            state[6, :, :] = 1.0

        return state

    # ── Memory ───────────────────────────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    # ── Training ─────────────────────────────────────────────────────────────
    def train_long_memory(self):
        """从 PER 池批量采样并训练，同步更新经验优先级"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        mini_sample, tree_idxs, is_weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        loss, td_errors = self.trainer.train_step(
            states, actions, rewards, next_states, dones, is_weights=is_weights
        )
        self.memory.update_priorities(tree_idxs, td_errors)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        """单步在线训练（短期记忆），不使用 IS 权重，不更新经验池优先级"""
        loss, _ = self.trainer.train_step(state, action, reward, next_state, done)
        return loss

    # ── Action ───────────────────────────────────────────────────────────────
    def get_action(self, state):
        """
        ε-greedy 探索策略：
          前 5000 局线性衰减（100% -> 1%），之后锁定在 1% 维持少量随机探索。

        修复：原代码用 random.randint(0, 100) 生成 0~100 共 101 个整数，
        导致概率计算偏差约 1%。改为 random.random() 生成 [0,1) 均匀浮点数，
        与 epsilon/100 比较，确保概率精确。
        """
        self.epsilon = max(1, 100 - (self.n_games / 50))

        final_move = [0, 0, 0]
        if random.random() * 100 < self.epsilon:
            # 探索：随机动作
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # 利用：模型预测最优动作
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


# ── Train Loop ───────────────────────────────────────────────────────────────
def train():
    agent = Agent()
    game  = SnakeGameAI()
    # 【新增】步数计数器
    total_steps = 0
    record = agent.load_checkpoint('checkpoint.pth')

    run_name = f"Snake_CNN_PER_{int(time.time())}"
    writer   = SummaryWriter(f'logs/{run_name}')
    print(f"📈 实时监控已开启，请在终端输入: tensorboard --logdir=logs")

    current_score = 0
    long_loss = 0.0  # 初始化 loss 用于打印

    try:
        while True:
            total_steps += 1  # 【新增】
            state_old  = agent.get_state(game)
            final_move = agent.get_action(state_old)

            # 每 50 局或高分时开启渲染，其余时间全速训练
            show_screen = (agent.n_games % 50 == 0) or (current_score > 30)
            reward, done, score = game.play_step(final_move, render=show_screen)
            current_score = score

            state_new  = agent.get_state(game)

            # 【核心修改 1】只记不练！彻底干掉单步在线训练 (train_short_memory)
            agent.remember(state_old, final_move, reward, state_new, done)


            if total_steps % 4 == 0:
                step_loss = agent.train_long_memory()
                if step_loss != 0.0:
                    long_loss = step_loss  # 更新最新的 loss 用于打印

            if done:
                game.reset()
                agent.n_games += 1

                # （这里原本的 agent.train_long_memory() 已经被我们挪到上面去了）

                if score > record:
                    record = score
                    agent.save_checkpoint(record, 'best_model.pth')
                    print("🏆 破纪录了！已单独保存 Best Model。")

                if agent.n_games % 50 == 0:
                    agent.save_checkpoint(record, 'checkpoint.pth')

                print(
                    f'Game {agent.n_games:>5} | '
                    f'Epsilon: {agent.epsilon:5.1f}% | '
                    f'Score: {score:>3} | '
                    f'Record: {record:>3} | '
                    f'Loss: {long_loss:.6f}'
                )

                writer.add_scalar('Loss/Training',    long_loss,        agent.n_games)
                writer.add_scalar('Score/Current',    score,            agent.n_games)
                writer.add_scalar('Score/Record',     record,           agent.n_games)
                writer.add_scalar('Metrics/Epsilon',  agent.epsilon,    agent.n_games)

                current_score = 0

    except KeyboardInterrupt:
        print("\n🛑 接收到手动停止信号 (Ctrl+C)，正在保存最后的训练现场...")
        agent.save_checkpoint(record, 'checkpoint.pth')
        print("✅ 存档成功！你可以安全关闭了，下次运行会自动接上进度。")
    finally:
        writer.close()


if __name__ == '__main__':
    train()
