from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MODEL_DIR = "./model"


def get_device() -> torch.device:
    """选择当前机器上最合适的 PyTorch 设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_dir(path: str = MODEL_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def _is_single_state(state: object) -> bool:
    if not isinstance(state, tuple) or len(state) != 2:
        return False
    image, direction = state
    return hasattr(image, "shape") and hasattr(direction, "shape")


def _state_to_tensors(state, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """把单条状态或状态批次统一转换成双模态 tensor。"""
    if _is_single_state(state):
        images, directions = state
    else:
        images, directions = zip(*state)

    image_array = np.asarray(images, dtype=np.float32)
    direction_array = np.asarray(directions, dtype=np.float32)

    if image_array.ndim == 3:
        image_array = np.expand_dims(image_array, axis=0)
    if direction_array.ndim == 1:
        direction_array = np.expand_dims(direction_array, axis=0)

    if image_array.ndim != 4:
        raise ValueError(f"图像状态应为 4 维 batch 张量，实际形状: {image_array.shape}")
    if direction_array.ndim != 2:
        raise ValueError(f"方向状态应为 2 维 batch 张量，实际形状: {direction_array.shape}")

    image_tensor = torch.as_tensor(image_array, dtype=torch.float32, device=device)
    direction_tensor = torch.as_tensor(direction_array, dtype=torch.float32, device=device)
    return image_tensor, direction_tensor


class Conv_QNet(nn.Module):
    """双模态 Q 网络：CNN 处理空间图，MLP 处理方向向量。"""

    def __init__(
        self,
        image_channels: int = 3,
        feature_size: int = 11,
        grid_h: int = 24,
        grid_w: int = 32,
        hidden_size: int = 256,
        output_size: int = 3,
    ):
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, grid_h, grid_w)
            vision_size = self.vision(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(vision_size + 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if features.ndim == 1:
            features = features.unsqueeze(0)

        vision_features = self.vision(image).flatten(1)
        vector_features = self.feature_encoder(features)
        merged_features = torch.cat((vision_features, vector_features), dim=1)
        return self.head(merged_features)

    def save(self, file_name: str = "model.pth") -> None:
        _ensure_dir()
        torch.save(self.state_dict(), os.path.join(MODEL_DIR, file_name))

    def load(self, file_name: str = "model.pth", device: torch.device | None = None) -> bool:
        path = os.path.join(MODEL_DIR, file_name)
        if not os.path.exists(path):
            return False

        map_location = device if device is not None else get_device()
        self.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
        return True


class QTrainer:
    def __init__(
        self,
        model: Conv_QNet,
        target_model: Conv_QNet,
        lr: float,
        gamma: float,
        device: torch.device | None = None,
        tau: float = 0.005,
        max_grad_norm: float = 10.0,
    ):
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.device = device if device is not None else get_device()
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self._amp_enabled = self.device.type == "cuda"
        self._amp_device = "cuda" if self._amp_enabled else "cpu"
        self.scaler = torch.amp.GradScaler(self._amp_device, enabled=self._amp_enabled)

    def train_step(self, state, action, reward, next_state, done, is_weights=None) -> tuple[float, np.ndarray]:
        image, direction = _state_to_tensors(state, self.device)
        next_image, next_direction = _state_to_tensors(next_state, self.device)

        action_tensor = torch.as_tensor(np.asarray(action), dtype=torch.long, device=self.device)
        reward_tensor = torch.as_tensor(np.asarray(reward), dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(np.asarray(done), dtype=torch.bool, device=self.device)

        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if reward_tensor.ndim == 0:
            reward_tensor = reward_tensor.unsqueeze(0)
        if done_tensor.ndim == 0:
            done_tensor = done_tensor.unsqueeze(0)
        if action_tensor.shape[1] != 3:
            raise ValueError(f"动作必须是 3 维 one-hot，实际形状: {tuple(action_tensor.shape)}")

        if is_weights is None:
            is_weight_tensor = torch.ones_like(reward_tensor, dtype=torch.float32, device=self.device)
        else:
            is_weight_tensor = torch.as_tensor(
                np.asarray(is_weights), dtype=torch.float32, device=self.device
            )
            if is_weight_tensor.ndim == 0:
                is_weight_tensor = is_weight_tensor.unsqueeze(0)

        with torch.amp.autocast(device_type=self._amp_device, enabled=self._amp_enabled):
            pred = self.model(image, direction)

            with torch.no_grad():
                next_q_online = self.model(next_image, next_direction)
                best_actions = torch.argmax(next_q_online, dim=1)
                next_q_target = self.target_model(next_image, next_direction)
                max_next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                target_q = reward_tensor + (~done_tensor).float() * self.gamma * max_next_q

            action_indices = torch.argmax(action_tensor, dim=1)
            current_q = pred.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
            elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
            loss = (is_weight_tensor * elementwise_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._soft_update_target()

        return float(loss.item()), td_errors

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.mul_(1.0 - self.tau).add_(model_param, alpha=self.tau)
