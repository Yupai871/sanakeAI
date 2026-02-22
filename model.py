import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用: {device} 进行训练")

MODEL_DIR = './model'


def _ensure_dir():
    """确保模型保存目录存在"""
    os.makedirs(MODEL_DIR, exist_ok=True)


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        _ensure_dir()
        path = os.path.join(MODEL_DIR, file_name)
        torch.save(self.state_dict(), path)

    def load(self, file_name='model.pth'):
        path = os.path.join(MODEL_DIR, file_name)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))
            print(f"🎉 成功加载前世记忆: {path}")
        else:
            print("👶 这是一个全新的大脑，开始从零学习！")


class Conv_QNet(nn.Module):
    def __init__(self, in_channels=7, grid_h=24, grid_w=32, output_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,          out_channels=32, kernel_size=4, stride=2, padding=1)

        # 修复1：动态推断 fc1 输入尺寸，不再依赖手工计算的魔法数字 32*6*8
        # 修改卷积参数或游戏分辨率后，此处永远自动正确，不会产生尺寸不匹配的运行时错误
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_h, grid_w)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            fc_in = dummy.numel()

        self.fc1 = nn.Linear(fc_in, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, file_name='model.pth'):
        # 修复2：复用公共辅助函数，消除与 Linear_QNet 的重复代码
        _ensure_dir()
        path = os.path.join(MODEL_DIR, file_name)
        torch.save(self.state_dict(), path)

    def load(self, file_name='model.pth'):
        path = os.path.join(MODEL_DIR, file_name)
        if os.path.exists(path):
            # 修复3：补齐 weights_only=True，与 Linear_QNet 保持一致，消除 PyTorch 2.x FutureWarning
            self.load_state_dict(torch.load(path, weights_only=True))
            print(f"🎉 成功加载前世记忆: {path}")
        else:
            print("👶 这是一个全新的 CNN 大脑，开始从零学习！")


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr           = lr
        self.gamma        = gamma
        self.model        = model
        self.target_model = target_model
        self.optimizer    = optim.Adam(model.parameters(), lr=self.lr)
        # 修复4：删除死代码 self.criterion = nn.MSELoss()
        # train_step 实际使用带 IS 权重的手写 MSE Loss，self.criterion 从未被调用
        self.tau          = 0.005

        # 修复5：根据实际设备动态创建 GradScaler
        # 原始代码硬编码 'cuda'，在 CPU / MPS 机器上直接崩溃
        # enabled=False 时 scaler 透明地退化为普通反向传播，代码路径完全统一
        self._amp_enabled = (device.type == 'cuda')
        self.scaler       = torch.amp.GradScaler(device.type, enabled=self._amp_enabled)

    def train_step(self, state, action, reward, next_state, done, is_weights=None):
        state      = torch.tensor(np.array(state),      dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action     = torch.tensor(action,               dtype=torch.long).to(device)
        reward     = torch.tensor(reward,               dtype=torch.float).to(device)
        done       = torch.tensor(done,                 dtype=torch.bool).to(device)

        # 单步调用时补充 batch 维度
        if len(state.shape) == 3:
            state      = torch.unsqueeze(state,      0)
            next_state = torch.unsqueeze(next_state, 0)
            action     = torch.unsqueeze(action,     0)
            reward     = torch.unsqueeze(reward,     0)
            done       = torch.unsqueeze(done,       0)

        # 没有传入权重（短期记忆单步训练）时，默认权重全为 1
        if is_weights is None:
            is_weights = torch.ones(len(done), dtype=torch.float).to(device)
        else:
            is_weights = torch.tensor(is_weights, dtype=torch.float).to(device)

        # 修复6：autocast 使用动态设备类型，CPU 上 enabled=False 自动跳过混合精度
        with torch.amp.autocast(device_type=device.type, enabled=self._amp_enabled):

            # 1. 预测当前状态下所有动作的 Q 值 (shape: Batch × 3)
            pred = self.model(state)

            # 2. Double DQN：online 网络选动作，target 网络估值，防止 Q 值过估计
            with torch.no_grad():
                next_q_online = self.model(next_state)
                best_actions  = torch.argmax(next_q_online, dim=1)
                next_q_target = self.target_model(next_state)
                max_next_q    = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

                # Bellman 目标值 (shape: Batch,)
                Q_new = reward + (~done).float() * self.gamma * max_next_q

            # 3. 用 gather 精准提取实际执行动作对应的预测 Q 值
            # one-hot 转索引：[0, 1, 0] -> 1
            action_indices = torch.argmax(action, dim=1)
            q_acted = pred.gather(1, action_indices.unsqueeze(1)).squeeze(1)

            # 4. 计算 TD-Error，用于更新 PER 优先级（detach，不参与反向传播）
            td_errors = torch.abs(Q_new - q_acted).detach().cpu().numpy()

            # 5. 带重要性采样权重的加权 MSE Loss
            # Q_new 无梯度，q_acted 保留计算图，反向传播方向正确
            loss = (is_weights * (Q_new - q_acted) ** 2).mean()

        # 反向传播（scaler 在 CPU 上 enabled=False 时透明地直接调用 backward）
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Soft Target Network Update: θ_target ← τ·θ_online + (1-τ)·θ_target
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item(), td_errors
