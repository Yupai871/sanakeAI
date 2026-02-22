import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font(None, 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# 颜色和尺寸设置
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)
BLACK = (0, 0, 0)
BLOCK_SIZE = 20


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('AI Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # 初始状态
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        # 修复：用迭代替代递归，避免蛇体极长时的栈溢出风险
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action, render=True):
        self.frame_iteration += 1

        # 1. 收集用户输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 记录移动前，蛇头到食物的曼哈顿距离
        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        # 2. 移动蛇的头部
        self._move(action)
        self.snake.insert(0, self.head)

        # 记录移动后，蛇头到食物的曼哈顿距离
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        # 3. 检查是否游戏结束
        reward = 0
        game_over = False
        # 如果撞墙、撞自己，或者原地绕圈太久没吃到食物
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # 死亡惩罚
            return reward, game_over, self.score

        # 4. 是否吃到食物
        if self.head == self.food:
            self.score += 1
            reward = 10  # 吃到食物的绝对核心奖励
            self._place_food()
        else:
            self.snake.pop()  # 没吃到食物，尾巴缩进一格

            # 修复：缩进对齐到 else 块内，奖励塑形逻辑现在会正确执行
            # 基于曼哈顿距离的连续奖励塑形
            # 网格最大距离约为 32 + 24 = 56
            # 距离奖励极小（±0.02），就算绕路 50 步最大惩罚也只有 -1.0，
            # 绝对不会覆盖 +10 的食物奖励
            if new_distance < old_distance:
                reward = 0.02   # 鼓励靠近
            elif new_distance > old_distance:
                reward = -0.02  # 远离惩罚，防止反复横跳刷分
            else:
                reward = 0      # 距离不变时不奖不罚

        # 5. 更新界面
        if render:
            self._update_ui()
            self.clock.tick(40)  # 只有需要看画面时才开启限速

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 1. 撞墙
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 2. 撞自己
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        # 画蛇
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        # 画食物
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # 画分数
        text = font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action = [直走, 右转, 左转] -> [1,0,0] 或 [0,1,0] 或 [0,0,1]

        # 建立一个顺时针的方向数组：右 -> 下 -> 左 -> 上
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # 获取当前方向在数组中的索引
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]           # 直走：方向不变
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]      # 右转：顺时针下一个方向
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]      # 左转：逆时针上一个方向

        # 更新蛇头方向
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
