import os
import random
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from agent import Agent, TrainConfig
from game import BLOCK_SIZE, Direction, Point, SnakeGameAI
from memory import PrioritizedReplayBuffer
from model import Conv_QNet, QTrainer


class SnakeAITests(unittest.TestCase):
    def make_agent(self):
        return Agent(
            TrainConfig(
                max_memory=64,
                batch_size=8,
                warmup_steps=8,
                train_every_steps=1,
                seed=123,
            ),
            device=torch.device("cpu"),
        )

    def test_state_shape_and_direction(self):
        game = SnakeGameAI()
        agent = self.make_agent()

        image, features = agent.get_state(game)

        self.assertEqual(image.shape, (3, 24, 32))
        self.assertEqual(features.shape, (11,))
        self.assertEqual(features[3:7].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(float(image[0].sum()), 1.0)
        self.assertEqual(float(image[1].sum()), 1.0)
        self.assertGreater(float(image[2].sum()), 0.0)

    def test_agent_action_remember_and_train_step(self):
        game = SnakeGameAI()
        agent = self.make_agent()
        state = agent.get_state(game)

        agent.n_games = agent.config.epsilon_decay_games
        action = agent.get_action(state)
        reward, done, _ = game.play_step(action, render=False)
        next_state = agent.get_state(game)
        agent.remember(state, action, reward, next_state, done)

        self.assertEqual(len(agent.memory), 1)
        loss = agent.train_short_memory(state, action, reward, next_state, done)
        self.assertTrue(np.isfinite(loss))

    def test_long_memory_updates_priorities(self):
        game = SnakeGameAI()
        agent = self.make_agent()

        for _ in range(agent.config.warmup_steps):
            state = agent.get_state(game)
            action = [1, 0, 0]
            reward, done, _ = game.play_step(action, render=False)
            next_state = agent.get_state(game)
            agent.remember(state, action, reward, next_state, done)
            if done:
                game.reset()

        loss = agent.train_long_memory()
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(agent.memory.tree.total_priority, 0)

    def test_model_batch_forward_and_trainer(self):
        model = Conv_QNet()
        target_model = Conv_QNet()
        target_model.load_state_dict(model.state_dict())
        trainer = QTrainer(model, target_model, lr=0.001, gamma=0.9, device=torch.device("cpu"))

        state = (
            np.zeros((3, 24, 32), dtype=np.float32),
            np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32),
        )
        next_state = (
            np.zeros((3, 24, 32), dtype=np.float32),
            np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0], dtype=np.float32),
        )
        loss, td_errors = trainer.train_step(
            [state, state],
            [[1, 0, 0], [0, 1, 0]],
            [0.02, -0.02],
            [next_state, next_state],
            [False, True],
            is_weights=[1.0, 0.5],
        )

        self.assertTrue(np.isfinite(loss))
        self.assertEqual(td_errors.shape, (2,))

    def test_per_handles_invalid_priorities(self):
        buffer = PrioritizedReplayBuffer(8)
        for index in range(4):
            buffer.add((index, index, index, index, False))

        sample, idxs, weights = buffer.sample(4)
        buffer.update_priorities(idxs, [float("nan"), float("inf"), -float("inf"), 0.0])

        self.assertEqual(len(sample), 4)
        self.assertEqual(weights.shape, (4,))
        self.assertTrue(np.isfinite(buffer.tree.total_priority))

    def test_food_generation_handles_full_board(self):
        game = SnakeGameAI(w=BLOCK_SIZE * 3, h=BLOCK_SIZE)
        game.snake = [Point(0, 0), Point(BLOCK_SIZE, 0), Point(BLOCK_SIZE * 2, 0)]
        game._place_food()
        self.assertIsNone(game.food)

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TrainConfig(max_memory=16, batch_size=4, warmup_steps=4, model_dir=tmp_dir, seed=7)
            agent = Agent(config=config, device=torch.device("cpu"))
            agent.n_games = 3
            agent.total_steps = 42
            agent.save_checkpoint(record=5)

            restored = Agent(config=config, device=torch.device("cpu"))
            record = restored.load_checkpoint()

            self.assertEqual(record, 5)
            self.assertEqual(restored.n_games, 3)
            self.assertEqual(restored.total_steps, 42)


if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    unittest.main()
