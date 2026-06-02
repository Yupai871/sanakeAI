from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from game import BLOCK_SIZE, Direction, Point, SnakeGameAI
from memory import PrioritizedReplayBuffer
from model import Conv_QNet, QTrainer, get_device


@dataclass(frozen=True)
class TrainConfig:
    max_memory: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 2_000
    train_every_steps: int = 4
    learning_rate: float = 0.0003
    gamma: float = 0.99
    epsilon_start: float = 100.0
    epsilon_end: float = 1.0
    epsilon_decay_games: int = 5_000
    render_every_games: int = 50
    render_score_threshold: int = 30
    checkpoint_every_games: int = 50
    model_dir: str = "./model"
    log_dir: str = "./logs"
    seed: int | None = None


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def create_summary_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"TensorBoard 不可用，训练将继续但不会写入日志: {exc}")
        return NullSummaryWriter()
    return SummaryWriter(log_dir)


class Agent:
    def __init__(self, config: TrainConfig | None = None, device: torch.device | None = None):
        self.config = config or TrainConfig()
        self.device = device if device is not None else get_device()
        self.n_games = 0
        self.total_steps = 0
        self.epsilon = self.config.epsilon_start

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        self.memory = PrioritizedReplayBuffer(capacity=self.config.max_memory)
        self.model = Conv_QNet().to(self.device)
        self.target_model = Conv_QNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(
            self.model,
            self.target_model,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            device=self.device,
        )

    def save_checkpoint(self, record: int, filename: str = "checkpoint.pth") -> None:
        os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(self.config.model_dir, filename)
        checkpoint = {
            "version": 2,
            "n_games": self.n_games,
            "total_steps": self.total_steps,
            "record": record,
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "optimizer_state": self.trainer.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str = "checkpoint.pth") -> int:
        path = os.path.join(self.config.model_dir, filename)
        if not os.path.exists(path):
            print("未找到 checkpoint，将从零开始训练。")
            return 0

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state"])
            self.target_model.load_state_dict(checkpoint["target_model_state"])
            self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        except RuntimeError as exc:
            print(f"checkpoint 与当前模型结构不兼容，已跳过加载: {exc}")
            return 0

        self.n_games = int(checkpoint.get("n_games", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        record = int(checkpoint.get("record", 0))
        print(f"已恢复训练现场：局数 {self.n_games}，步数 {self.total_steps}，最高分 {record}")
        return record

    def get_state(self, game: SnakeGameAI) -> tuple[np.ndarray, np.ndarray]:
        grid_w = game.w // BLOCK_SIZE
        grid_h = game.h // BLOCK_SIZE
        image_state = np.zeros((3, grid_h, grid_w), dtype=np.float32)

        food_x, food_y = self._point_to_grid(game.food)
        if 0 <= food_x < grid_w and 0 <= food_y < grid_h:
            image_state[0, food_y, food_x] = 1.0

        head_x, head_y = self._point_to_grid(game.head)
        if 0 <= head_x < grid_w and 0 <= head_y < grid_h:
            image_state[1, head_y, head_x] = 1.0

        body_len = len(game.snake) - 1
        for index, point in enumerate(game.snake[1:]):
            body_x, body_y = self._point_to_grid(point)
            if 0 <= body_x < grid_w and 0 <= body_y < grid_h:
                image_state[2, body_y, body_x] = self._body_intensity(index, body_len)

        return image_state, self._vector_features(game)

    def get_action(self, state: tuple[np.ndarray, np.ndarray]) -> list[int]:
        self.epsilon = self._current_epsilon()
        move = random.randint(0, 2)

        if random.random() * 100 >= self.epsilon:
            image, features = state
            image_tensor = torch.as_tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0)
            feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(image_tensor, feature_tensor)
            if was_training:
                self.model.train()
            move = int(torch.argmax(prediction).item())

        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.add((state, action, reward, next_state, done))

    def train_long_memory(self) -> float:
        if len(self.memory) < self.config.warmup_steps:
            return 0.0

        sample_size = min(self.config.batch_size, len(self.memory))
        mini_sample, tree_idxs, is_weights = self.memory.sample(sample_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss, td_errors = self.trainer.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones,
            is_weights=is_weights,
        )
        self.memory.update_priorities(tree_idxs, td_errors)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done) -> float:
        loss, _ = self.trainer.train_step(state, action, reward, next_state, done)
        return loss

    def _current_epsilon(self) -> float:
        progress = min(self.n_games / self.config.epsilon_decay_games, 1.0)
        span = self.config.epsilon_start - self.config.epsilon_end
        return self.config.epsilon_start - span * progress

    @staticmethod
    def _point_to_grid(point) -> tuple[int, int]:
        return int(point.x // BLOCK_SIZE), int(point.y // BLOCK_SIZE)

    @staticmethod
    def _body_intensity(index: int, body_len: int) -> float:
        if body_len <= 1:
            return 1.0
        return 1.0 - 0.8 * (index / (body_len - 1))

    def _vector_features(self, game: SnakeGameAI) -> np.ndarray:
        head = game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        danger_straight = (
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d))
        )
        danger_right = (
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d))
        )
        danger_left = (
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d))
        )

        return np.array(
            [
                danger_straight,
                danger_right,
                danger_left,
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                game.food.x < head.x,
                game.food.x > head.x,
                game.food.y < head.y,
                game.food.y > head.y,
            ],
            dtype=np.float32,
        )


def train(config: TrainConfig | None = None) -> None:
    config = config or TrainConfig()
    agent = Agent(config=config)
    print(f"正在使用 {agent.device} 训练")

    game = SnakeGameAI()
    record = agent.load_checkpoint("checkpoint.pth")
    writer = create_summary_writer(os.path.join(config.log_dir, f"Snake_CNN_PER_{int(time.time())}"))
    print("训练已启动；TensorBoard 命令: tensorboard --logdir=logs")

    current_score = 0
    long_loss = 0.0

    try:
        while True:
            agent.total_steps += 1
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)

            show_screen = (
                agent.n_games % config.render_every_games == 0
                or current_score > config.render_score_threshold
            )
            reward, done, score = game.play_step(final_move, render=show_screen)
            current_score = score
            state_new = agent.get_state(game)
            agent.remember(state_old, final_move, reward, state_new, done)

            if agent.total_steps % config.train_every_steps == 0:
                step_loss = agent.train_long_memory()
                if step_loss:
                    long_loss = step_loss

            if done:
                game.reset()
                agent.n_games += 1

                if score > record:
                    record = score
                    agent.save_checkpoint(record, "best_model.pth")
                    print("刷新最高分，已保存 best_model.pth")

                if agent.n_games % config.checkpoint_every_games == 0:
                    agent.save_checkpoint(record, "checkpoint.pth")

                print(
                    f"Game {agent.n_games:>5} | "
                    f"Steps {agent.total_steps:>8} | "
                    f"Epsilon {agent.epsilon:5.1f}% | "
                    f"Score {score:>3} | "
                    f"Record {record:>3} | "
                    f"Loss {long_loss:.6f}"
                )

                writer.add_scalar("Loss/Training", long_loss, agent.n_games)
                writer.add_scalar("Score/Current", score, agent.n_games)
                writer.add_scalar("Score/Record", record, agent.n_games)
                writer.add_scalar("Metrics/Epsilon", agent.epsilon, agent.n_games)
                current_score = 0

    except KeyboardInterrupt:
        print("\n收到停止信号，正在保存 checkpoint...")
        agent.save_checkpoint(record, "checkpoint.pth")
        print("保存完成。")
    finally:
        writer.close()


def play(config: TrainConfig | None = None, model_file: str = "best_model.pth", fps: int = 30) -> None:
    config = config or TrainConfig()
    agent = Agent(config=config)
    record = agent.load_checkpoint(model_file)
    agent.n_games = agent.config.epsilon_decay_games
    game = SnakeGameAI()
    print(f"已加载 {os.path.join(config.model_dir, model_file)}，训练最高分 {record}。按关闭窗口退出。")
    episode_steps = 0
    run_id = 1

    while True:
        episode_steps += 1
        state = agent.get_state(game)
        action = agent.get_action(state)
        _, done, score = game.play_step(action, render=True)
        game.clock.tick(fps)

        if done:
            print(f"Run {run_id} | Score {score} | Steps {episode_steps}")
            game.reset()
            episode_steps = 0
            run_id += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake AI train/play entrypoint")
    parser.add_argument(
        "--play",
        action="store_true",
        help="加载训练好的模型，在 Pygame 可视化窗口中运行",
    )
    parser.add_argument(
        "--model-dir",
        default="./model",
        help="模型目录，例如 ./model_v2",
    )
    parser.add_argument(
        "--model-file",
        default="best_model.pth",
        help="模型文件名，默认 best_model.pth",
    )
    parser.add_argument(
        "--log-dir",
        default="./logs",
        help="训练日志目录",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="可视化运行帧率",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runtime_config = TrainConfig(model_dir=args.model_dir, log_dir=args.log_dir)
    if args.play:
        play(runtime_config, model_file=args.model_file, fps=args.fps)
    else:
        train(runtime_config)
