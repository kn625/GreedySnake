# Snake AI: Deep Q-Network Based Snake Game

## 项目概述

Snake AI是一个基于深度Q网络(DQN)的智能蛇形游戏项目，旨在通过强化学习训练AI代理自主玩经典的贪吃蛇游戏。该项目特别解决了长蛇情况下频繁发生的自碰撞问题，通过创新的状态表示方法使AI能够更好地感知自身身体状态，从而提升了模型在复杂环境下的性能稳定性。

## 核心功能

- 🎮 **经典贪吃蛇游戏**：使用Pygame实现的完整贪吃蛇游戏逻辑
- 🧠 **DQN智能代理**：基于深度Q网络的强化学习AI，能够自主学习和提高游戏技能
- 👁️ **高级状态感知**：创新的状态表示方法，包含蛇身部位信息，有效解决长蛇自碰撞问题
- 📊 **实时训练可视化**：训练过程中提供分数、损失、探索率等关键指标的实时图表
- 🎯 **多模式支持**：提供训练模式、自动玩模式和手动模式
- 💾 **模型持久化**：支持保存和加载训练好的模型权重

## 技术栈

| 技术/库 | 用途 | 版本要求 |
|--------|------|---------|
| Python | 主要开发语言 | 3.6+ |
| Pygame | 游戏渲染和用户交互 | 2.0+ |
| PyTorch | 深度学习模型实现 | 1.8+ |
| NumPy | 数值计算和数组操作 | 1.18+ |
| Matplotlib | 训练可视化和图表绘制 | 3.3+ |

## 环境配置要求

- 操作系统：Windows、macOS或Linux
- Python版本：3.6及以上
- 建议使用GPU加速（可选，用于加速模型训练）

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd Snake_AI
```

### 2. 创建虚拟环境（可选但推荐）

```bash
# 使用venv创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install pygame torch numpy matplotlib
```

## 使用指南

### 训练模式

训练模式用于训练DQN模型，支持实时可视化监控训练过程。

```bash
python greedy_snake.py
```

默认情况下，程序会直接进入训练模式，开始训练AI代理。训练过程中会显示以下信息：
- 每轮游戏的分数
- 训练损失值
- 探索率(ε)
- 平均分数和平均损失

### 自动玩模式

使用训练好的模型自动玩游戏。

```bash
python greedy_snake.py --play --model snake_dqn_model_final.pth
```

### 手动模式

手动控制蛇的移动，使用方向键控制蛇的上下左右移动。

```bash
python greedy_snake.py --manual
```

## 核心实现详解

### 游戏核心类 (GameCore)

GameCore类封装了所有游戏逻辑，包括蛇的移动、食物生成、碰撞检测等。

#### 主要方法

- `reset()`: 重置游戏状态
- `step(action)`: 执行一步游戏，返回新状态、奖励和是否结束
- `render()`: 渲染游戏画面
- `get_state()`: 获取游戏状态（用于AI训练）

### DQN智能代理 (DQNAgent)

DQNAgent类实现了深度Q网络算法，用于训练AI代理。

#### 主要方法

- `act(state)`: 根据当前状态选择动作
- `train(batch_size)`: 训练模型
- `save_model(file_name)`: 保存模型权重
- `load_model(file_name)`: 加载模型权重

### 状态表示

状态向量包含42个特征，主要包括：
- 移动方向的one-hot编码
- 蛇头周围8个方向的危险检测
- 扩展危险检测（前方2-3个方块）
- 食物相对位置
- 蛇身弯折数量
- 最近身体部位的距离和方向
- 蛇身关键部位的相对位置
- 蛇身形状特征

### 解决长蛇自碰撞问题的关键改进

1. **扩展危险检测**：检测蛇头前方2-3个方块是否有身体部位
2. **蛇身弯折数量**：实时跟踪蛇身体的弯折点数量
3. **最近身体部位信息**：计算蛇头到最近身体部位的距离和方向
4. **蛇身关键部位相对位置**：提取最近5个身体部位的相对坐标
5. **蛇身形状特征**：计算并归一化蛇身占据的宽度和高度

## API文档

### GameCore类

```python
class GameCore:
    def __init__(self, width=800, height=600, block_size=10, speed=200):
        # 初始化游戏窗口、蛇的位置、食物位置等
    
    def reset(self):
        # 重置游戏状态，返回初始状态
    
    def step(self, action):
        # 执行动作，返回(next_state, reward, done)
    
    def render(self):
        # 渲染游戏画面
    
    def get_state(self):
        # 获取游戏状态，返回状态向量
```

### DQNAgent类

```python
class DQNAgent:
    def __init__(self, state_size, action_size):
        # 初始化DQN模型、经验回放缓冲区等
    
    def act(self, state):
        # 根据状态选择动作
    
    def remember(self, state, action, reward, next_state, done):
        # 保存经验到回放缓冲区
    
    def train(self, batch_size):
        # 训练模型，返回损失值
    
    def save_model(self, file_name):
        # 保存模型权重
    
    def load_model(self, file_name):
        # 加载模型权重
```

## 更新日志

### v1.0.0 (2025-12-28)
- 初始版本发布
- 实现了完整的贪吃蛇游戏逻辑
- 基于DQN的AI代理
- 解决了长蛇自碰撞问题
- 支持训练、自动玩和手动模式

---

享受游戏和AI训练的乐趣！🐍🤖