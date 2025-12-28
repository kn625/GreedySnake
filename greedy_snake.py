import pygame
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# 初始化 Pygame
pygame.init()

# 定义颜色
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# 游戏核心类，封装所有游戏逻辑
class GameCore:
    def __init__(self, width=800, height=600, block_size=10, speed=200):
        self.dis_width = width
        self.dis_height = height
        self.snake_block = block_size
        self.snake_speed = speed
        
        # 创建游戏窗口
        self.dis = pygame.display.set_mode((self.dis_width, self.dis_height))
        pygame.display.set_caption('Snake Game')
        
        # 定义时钟对象
        self.clock = pygame.time.Clock()
        
        # 字体设置
        self.font_style = pygame.font.SysFont("Fira Code", 25)
        self.score_font = pygame.font.SysFont("Fira Code", 35)
        
        # 初始化游戏状态
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        # 蛇的初始位置
        self.x1 = self.dis_width / 2
        self.y1 = self.dis_height / 2
        
        self.x1_change = 0
        self.y1_change = 0
        
        self.snake_List = []
        self.Length_of_snake = 1
        
        # 食物的随机位置
        self.foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
        self.foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0
        
        # 初始方向和距离
        self.current_direction = 0  # 初始方向设为向左
        self.previous_direction = 0  # 记录前一个方向
        self.distance_old = np.sqrt((self.x1 - self.foodx)**2 + (self.y1 - self.foody)**2)
        self.turns = []  # 记录所有拐弯点的位置
        
        self.game_over = False
        self.game_close = False
        
        return self.get_state()
    
    def step(self, action):
        """执行一步游戏，返回新状态、奖励、是否结束"""
        # 记录前一个方向
        self.previous_direction = self.current_direction
        
        # 更新当前方向
        if action == 0 and self.x1_change != self.snake_block:
            self.x1_change = -self.snake_block
            self.y1_change = 0
            self.current_direction = 0  # 左
        elif action == 1 and self.x1_change != -self.snake_block:
            self.x1_change = self.snake_block
            self.y1_change = 0
            self.current_direction = 1  # 右
        elif action == 2 and self.y1_change != self.snake_block:
            self.y1_change = -self.snake_block
            self.x1_change = 0
            self.current_direction = 2  # 上
        elif action == 3 and self.y1_change != -self.snake_block:
            self.y1_change = self.snake_block
            self.x1_change = 0
            self.current_direction = 3  # 下
        
        # 更新位置
        self.x1 += self.x1_change
        self.y1 += self.y1_change
        
        # 碰撞检测
        done = False
        reward = 0
        
        # 边界碰撞
        if self.x1 >= self.dis_width or self.x1 < 0 or self.y1 >= self.dis_height or self.y1 < 0:
            done = True
            reward = -100  # 边界碰撞惩罚
        
        # 自身碰撞
        snake_Head = []
        snake_Head.append(self.x1)
        snake_Head.append(self.y1)
        self.snake_List.append(snake_Head)
        
        # 检测是否发生拐弯
        if self.current_direction != self.previous_direction:
            # 记录当前位置为拐弯点
            turn_pos = [self.x1, self.y1]
            self.turns.append(turn_pos)
        
        old_tail_pos = None
        
        # 更新蛇身，移除旧的尾部（如果长度超过）
        if len(self.snake_List) > self.Length_of_snake:
            old_tail_pos = self.snake_List[0]  # 记录旧的尾部位置
            del self.snake_List[0]  # 移除旧的尾部
            
            # 检查蛇尾是否离开任何拐弯点
            if old_tail_pos is not None:
                # 检查旧的尾部位置是否在拐弯点列表中
                for i, turn_pos in enumerate(self.turns):
                    if turn_pos == old_tail_pos:
                        # 移除该拐弯点
                        del self.turns[i]
                        break
        
        for x in self.snake_List[:-1]:
            if x == snake_Head:
                done = True
                reward = -100  # 自身碰撞惩罚
        
        # 如果游戏未结束，计算食物相关奖励
        if not done:
            # 判断蛇是否吃到食物
            if self.x1 == self.foodx and self.y1 == self.foody:
                self.foodx = round(random.randrange(0, self.dis_width - self.snake_block) / 10.0) * 10.0
                self.foody = round(random.randrange(0, self.dis_height - self.snake_block) / 10.0) * 10.0
                self.Length_of_snake += 1
                reward = 50  # 吃到食物奖励
            else:
                # 计算与食物的距离变化，给予中间奖励
                distance_to_food = np.sqrt((self.x1 - self.foodx)**2 + (self.y1 - self.foody)**2)
                if distance_to_food < self.distance_old:
                    reward = 1  # 向食物移动给予小奖励
                else:
                    reward = -1  # 远离食物给予小惩罚
        
        # 记录当前距离用于下一次比较
        self.distance_old = np.sqrt((self.x1 - self.foodx)**2 + (self.y1 - self.foody)**2)
        
        # 获取新状态
        next_state = self.get_state()
        
        return next_state, reward, done
    
    def render(self):
        """渲染游戏画面"""
        self.dis.fill(blue)
        
        # 绘制食物
        pygame.draw.rect(self.dis, green, [self.foodx, self.foody, self.snake_block, self.snake_block])
        
        # 绘制蛇
        for x in self.snake_List:
            pygame.draw.rect(self.dis, black, [x[0], x[1], self.snake_block, self.snake_block])
        
        # 显示分数
        self.Your_score()
        
        pygame.display.update()
    
    def Your_score(self):
        """显示分数"""
        value = self.score_font.render("Your Score: " + str(self.Length_of_snake - 1), True, yellow)
        self.dis.blit(value, [0, 0])
    
    def message(self, msg, color):
        """显示游戏结束信息"""
        mesg = self.font_style.render(msg, True, color)
        self.dis.blit(mesg, [self.dis_width / 6, self.dis_height / 3])
    
    def get_state(self):
        """获取游戏状态"""
        if len(self.snake_List) == 0:
            return np.zeros(42)  # 状态维度从23增加到42
            
        snake_head = self.snake_List[-1]
        
        # 移动方向的one-hot编码
        direction_left = 1 if self.current_direction == 0 else 0
        direction_right = 1 if self.current_direction == 1 else 0
        direction_up = 1 if self.current_direction == 2 else 0
        direction_down = 1 if self.current_direction == 3 else 0
        
        # 蛇头周围8个方向是否有障碍物（边界或自身）
        head_x = snake_head[0]
        head_y = snake_head[1]
        
        danger_left = 1 if head_x - self.snake_block < 0 or [head_x - self.snake_block, head_y] in self.snake_List else 0
        danger_right = 1 if head_x + self.snake_block >= self.dis_width or [head_x + self.snake_block, head_y] in self.snake_List else 0
        danger_up = 1 if head_y - self.snake_block < 0 or [head_x, head_y - self.snake_block] in self.snake_List else 0
        danger_down = 1 if head_y + self.snake_block >= self.dis_height or [head_x, head_y + self.snake_block] in self.snake_List else 0
        danger_up_left = 1 if head_x - self.snake_block < 0 or head_y - self.snake_block < 0 or [head_x - self.snake_block, head_y - self.snake_block] in self.snake_List else 0
        danger_up_right = 1 if head_x + self.snake_block >= self.dis_width or head_y - self.snake_block < 0 or [head_x + self.snake_block, head_y - self.snake_block] in self.snake_List else 0
        danger_down_left = 1 if head_x - self.snake_block < 0 or head_y + self.snake_block >= self.dis_height or [head_x - self.snake_block, head_y + self.snake_block] in self.snake_List else 0
        danger_down_right = 1 if head_x + self.snake_block >= self.dis_width or head_y + self.snake_block >= self.dis_height or [head_x + self.snake_block, head_y + self.snake_block] in self.snake_List else 0
        
        # 扩展危险检测范围 - 检查蛇头前方2-3个方块
        danger_ahead_2 = 0
        danger_ahead_3 = 0
        if self.current_direction == 0:  # 左
            if head_x - 2 * self.snake_block >= 0 and [head_x - 2 * self.snake_block, head_y] in self.snake_List:
                danger_ahead_2 = 1
            if head_x - 3 * self.snake_block >= 0 and [head_x - 3 * self.snake_block, head_y] in self.snake_List:
                danger_ahead_3 = 1
        elif self.current_direction == 1:  # 右
            if head_x + 2 * self.snake_block < self.dis_width and [head_x + 2 * self.snake_block, head_y] in self.snake_List:
                danger_ahead_2 = 1
            if head_x + 3 * self.snake_block < self.dis_width and [head_x + 3 * self.snake_block, head_y] in self.snake_List:
                danger_ahead_3 = 1
        elif self.current_direction == 2:  # 上
            if head_y - 2 * self.snake_block >= 0 and [head_x, head_y - 2 * self.snake_block] in self.snake_List:
                danger_ahead_2 = 1
            if head_y - 3 * self.snake_block >= 0 and [head_x, head_y - 3 * self.snake_block] in self.snake_List:
                danger_ahead_3 = 1
        elif self.current_direction == 3:  # 下
            if head_y + 2 * self.snake_block < self.dis_height and [head_x, head_y + 2 * self.snake_block] in self.snake_List:
                danger_ahead_2 = 1
            if head_y + 3 * self.snake_block < self.dis_height and [head_x, head_y + 3 * self.snake_block] in self.snake_List:
                danger_ahead_3 = 1
        
        # 食物相对位置和方向
        food_left = 1 if self.foodx < head_x else 0
        food_right = 1 if self.foodx > head_x else 0
        food_up = 1 if self.foody < head_y else 0
        food_down = 1 if self.foody > head_y else 0
        
        # 蛇身体当前弯折数量（使用turns列表长度，归一化）
        normalized_bend_count = len(self.turns)**2 / len(self.snake_List) if len(self.snake_List) > 0 else 0
        
        # 计算蛇头到最近身体部位的距离和方向
        min_distance = float('inf')
        min_distance_dir = [0, 0, 0, 0]  # 左, 右, 上, 下
        
        # 遍历蛇身（跳过蛇头）
        for body_part in self.snake_List[:-1]:
            dx = body_part[0] - head_x
            dy = body_part[1] - head_y
            distance = abs(dx) + abs(dy)  # 曼哈顿距离
            
            if distance < min_distance:
                min_distance = distance
                # 确定最近身体部位的方向
                min_distance_dir = [0, 0, 0, 0]
                if abs(dx) > abs(dy):
                    if dx < 0:
                        min_distance_dir[0] = 1  # 左
                    else:
                        min_distance_dir[1] = 1  # 右
                else:
                    if dy < 0:
                        min_distance_dir[2] = 1  # 上
                    else:
                        min_distance_dir[3] = 1  # 下
        
        normalized_min_distance = min_distance / (self.dis_width + self.dis_height)
        
        # 获取蛇身关键部位的相对位置
        # 取最近的5个身体部位（如果有的话）
        body_parts_features = []
        max_body_parts = 5
        for i in range(1, min(max_body_parts + 1, len(self.snake_List))):
            body_part = self.snake_List[-1 - i]
            # 计算相对位置并归一化
            rel_x = (body_part[0] - head_x) / self.dis_width
            rel_y = (body_part[1] - head_y) / self.dis_height
            body_parts_features.extend([rel_x, rel_y])
        
        # 不足5个身体部位则用0填充
        while len(body_parts_features) < 2 * max_body_parts:
            body_parts_features.extend([0, 0])
        
        # 蛇身形状特征
        snake_body = self.snake_List[:-1]  # 排除蛇头
        
        # 计算蛇身占据的宽度和高度
        if len(snake_body) > 1:
            body_x = [part[0] for part in snake_body]
            body_y = [part[1] for part in snake_body]
            body_width = max(body_x) - min(body_x) + self.snake_block
            body_height = max(body_y) - min(body_y) + self.snake_block
            normalized_body_width = body_width / self.dis_width
            normalized_body_height = body_height / self.dis_height
        else:
            normalized_body_width = 0
            normalized_body_height = 0
        
        state = [
            # 移动方向
            direction_left, direction_right, direction_up, direction_down,
            # 周围危险情况
            danger_left, danger_right, danger_up, danger_down,
            danger_up_left, danger_up_right, danger_down_left, danger_down_right,
            # 扩展危险检测
            danger_ahead_2, danger_ahead_3,
            # 食物相对位置
            food_left, food_right, food_up, food_down,
            # 蛇头到食物的相对距离
            (self.foodx - head_x) / self.dis_width,
            (self.foody - head_y) / self.dis_height,
            # 蛇头到边界的距离
            head_x / self.dis_width,
            (self.dis_width - head_x) / self.dis_width,
            head_y / self.dis_height,
            (self.dis_height - head_y) / self.dis_height,
            # 蛇身体当前弯折数量（归一化）
            normalized_bend_count,
            # 最近身体部位信息
            *min_distance_dir,
            normalized_min_distance,
            # 蛇身关键部位相对位置
            *body_parts_features,
            # 蛇身形状特征
            normalized_body_width,
            normalized_body_height
        ]
        # 验证状态向量长度
        assert len(state) == 42, f"State vector length is {len(state)}, expected 42"
        
        # 检查状态向量中是否有NaN或无穷大值
        state_array = np.array(state)
        if np.isnan(state_array).any() or np.isinf(state_array).any():
            # 替换NaN和无穷大值为0
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state_array
    
    def handle_events(self):
        """处理事件，用于手动模式"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.x1_change != self.snake_block:
                    return 0  # 左
                elif event.key == pygame.K_RIGHT and self.x1_change != -self.snake_block:
                    return 1  # 右
                elif event.key == pygame.K_UP and self.y1_change != self.snake_block:
                    return 2  # 上
                elif event.key == pygame.K_DOWN and self.y1_change != -self.snake_block:
                    return 3  # 下
                elif event.key == pygame.K_q:
                    return "quit"
                elif event.key == pygame.K_c:
                    return "restart"
        return None
    
    def tick(self, speed=None):
        """控制游戏速度"""
        if speed is None:
            speed = self.snake_speed
        self.clock.tick(speed)


# 定义神经网络
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # 使用传入的hidden_size参数构建网络
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        
        # 自定义参数初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """使用更适合DQN的权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对于隐藏层使用Kaiming初始化
                if m.out_features != 4:  # 不是输出层
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                else:  # 输出层使用较小的初始化值
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                
                # 偏置初始化为0
                nn.init.zeros_(m.bias)

    def forward(self, x):
        l1 = torch.relu(self.fc1(x))
        l2 = torch.relu(self.fc2(l1))
        l3 = torch.relu(self.fc3(l2))
        l4 = torch.relu(self.fc4(l3))
        return self.fc5(l4)


# 定义 DQN 智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # 增大经验回放缓冲区
        self.gamma = 0.99  # 提高折扣因子，更重视长期奖励
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.001  # 降低最小探索率
        self.epsilon_decay = 0.995  # 加快探索率衰减
        self.learning_rate = 0.00001  # 降低学习率以提高稳定性
        self.model = DQN(state_size, 256, action_size)  # 隐藏层大小不再使用，保持参数兼容
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # 使用Huber损失(SmoothL1Loss)，提高对异常值的鲁棒性

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # 随机探索时排除危险动作
            # state结构：
            # 0-3: 方向信息 (left, right, up, down)
            # 4-11: 危险信息 (left, right, up, down, up_left, up_right, down_left, down_right)
            
            # 获取当前方向
            current_direction = np.argmax(state[0:4])
            
            # 获取危险信息
            danger_left = state[4]
            danger_right = state[5]
            danger_up = state[6]
            danger_down = state[7]
            
            # 确定安全动作列表
            safe_actions = []
            
            # 检查向左移动是否安全
            if not danger_left and current_direction != 1:  # 不能直接反向
                safe_actions.append(0)
            
            # 检查向右移动是否安全
            if not danger_right and current_direction != 0:  # 不能直接反向
                safe_actions.append(1)
            
            # 检查向上移动是否安全
            if not danger_up and current_direction != 3:  # 不能直接反向
                safe_actions.append(2)
            
            # 检查向下移动是否安全
            if not danger_down and current_direction != 2:  # 不能直接反向
                safe_actions.append(3)
            
            # 如果有安全动作，从安全动作中随机选择
            if safe_actions:
                return random.choice(safe_actions)
            else:
                # 理论上不应该发生，但为了防止错误，仍允许随机选择
                return random.randrange(self.action_size)
        
        # 利用模型选择动作
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        action = np.argmax(act_values.detach().numpy())
        return action

    def train(self, batch_size):
        '''
        从经验回放缓冲区中随机采样一个批次的经验，用于训练模型。
        '''
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # 批量处理：将所有样本合并为批次张量
        states = torch.FloatTensor([s for s, a, r, ns, d in minibatch])
        actions = torch.LongTensor([a for s, a, r, ns, d in minibatch])
        rewards = torch.FloatTensor([r for s, a, r, ns, d in minibatch])
        next_states = torch.FloatTensor([ns for s, a, r, ns, d in minibatch])
        dones = torch.BoolTensor([d for s, a, r, ns, d in minibatch])
        
        # 计算当前状态的Q值
        current_q = self.model(states)
        
        # 计算目标Q值
        next_q = self.model(next_states)
        max_next_q = torch.max(next_q, dim=1)[0]
        target_q = current_q.clone()
        
        # 更新目标Q值：使用masked_scatter更高效
        target_q[range(batch_size), actions] = rewards + (self.gamma * max_next_q * ~dones)
        
        # 计算损失并进行反向传播
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        self.optimizer.step()
        
        loss_total = loss.item() * batch_size  # 损失已经是平均值，乘以batch_size得到总和
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss_total / batch_size  # 返回平均损失
    
    def save_model(self, file_name):
        """保存模型到文件"""
        torch.save(self.model.state_dict(), file_name)
        
    def load_model(self, file_name):
        """从文件加载模型"""
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()


def train_mode(show_visualization=True):
    """训练模式，支持实时可视化监控"""
    game = GameCore()
    agent = DQNAgent(42, 4)
    batch_size = 128
    EPISODES = 2000
    
    # 用于跟踪训练过程的变量
    training_stats = {
        'episode': [],
        'score': [],
        'loss': [],
        'epsilon': [],
        'avg_score': [],
        'avg_loss': [],
        'total_reward': [],
        'avg_reward': []
    }
    
    # 用于计算移动平均值的窗口
    SCORE_WINDOW = 10
    LOSS_WINDOW = 100
    REWARD_WINDOW = 10  # 奖励平均值窗口
    
    # 用于存储最近的分数、损失和奖励
    recent_scores = deque(maxlen=SCORE_WINDOW)
    recent_losses = deque(maxlen=LOSS_WINDOW)
    recent_rewards = deque(maxlen=REWARD_WINDOW)
    
    # 初始化可视化
    if show_visualization:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))
        fig.tight_layout(pad=3.0)
        
        # 设置标题
        ax1.set_title('Score over Episodes')
        ax2.set_title('Loss over Steps')
        ax3.set_title('Epsilon over Episodes')
        ax4.set_title('Total Reward over Episodes')
        
        # 设置标签
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        
        # 初始化绘图数据
        score_line, = ax1.plot([], [], 'b-', label='Score')
        avg_score_line, = ax1.plot([], [], 'r-', label=f'Avg Score ({SCORE_WINDOW})')
        loss_line, = ax2.plot([], [], 'g-', label='Loss')
        avg_loss_line, = ax2.plot([], [], 'y-', label=f'Avg Loss ({LOSS_WINDOW})')
        epsilon_line, = ax3.plot([], [], 'm-', label='Epsilon')
        reward_line, = ax4.plot([], [], 'c-', label='Total Reward')
        avg_reward_line, = ax4.plot([], [], 'orange', label=f'Avg Reward ({REWARD_WINDOW})')
        
        # 添加图例
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        
        # 自动调整轴范围
        ax1.autoscale_view()
        ax2.autoscale_view()
        ax3.autoscale_view()
        ax4.autoscale_view()
        
        # 设置实时更新函数
        def update_plot(frame):
            # 更新分数图
            if training_stats['episode']:
                score_line.set_data(training_stats['episode'], training_stats['score'])
                if training_stats['avg_score']:
                    avg_score_line.set_data(training_stats['episode'], training_stats['avg_score'])
            
            # 更新损失图
            if training_stats['loss']:
                # 使用step_counter作为X轴坐标
                loss_steps = list(range(1, len(training_stats['loss']) + 1))
                loss_line.set_data(loss_steps, training_stats['loss'])
                
                # 更新移动平均损失图
                if training_stats['avg_loss']:
                    avg_loss_line.set_data(loss_steps[:len(training_stats['avg_loss'])], training_stats['avg_loss'])
            
            # 更新探索率图
            if training_stats['episode']:
                epsilon_line.set_data(training_stats['episode'], training_stats['epsilon'])
                
                # 更新奖励图
                reward_line.set_data(training_stats['episode'], training_stats['total_reward'])
                if training_stats['avg_reward']:
                    avg_reward_line.set_data(training_stats['episode'], training_stats['avg_reward'])
            
            # 重新计算轴范围
            for ax in [ax1, ax2, ax3, ax4]:
                ax.relim()
                ax.autoscale_view()
            
            return score_line, avg_score_line, loss_line, avg_loss_line, epsilon_line, reward_line, avg_reward_line
        
        plt.ion()  # 开启交互模式
        plt.show(block=False)
        plt.pause(0.1)  # 初始化显示
    
    step_counter = 0
    
    for e in range(EPISODES):
        state = game.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作，获取新状态、奖励和是否结束
            next_state, reward, done = game.step(action)
            episode_reward += reward
            
            # 记忆经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            
            # 渲染游戏
            game.render()
            
            # 训练模型
            if len(agent.memory) > batch_size:
                loss = agent.train(batch_size)
                step_counter += 1
                
                # 记录损失
                if loss is not None:
                    recent_losses.append(loss)
                    training_stats['loss'].append(loss)
                    
                    # 计算并记录平均损失
                    if recent_losses:
                        avg_loss = np.mean(recent_losses)
                        training_stats['avg_loss'].append(avg_loss)
                    
                    # 每50步打印一次详细信息
                    if step_counter % 100 == 0:
                        avg_loss = np.mean(recent_losses) if recent_losses else 0
                        print(f"[TRAINING] Episode: {e + 1:4d}, Cur Score: {game.Length_of_snake-1:3d}, Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")
            
            # 控制游戏速度
            game.tick()
        
        # 记录每轮信息
        score = game.Length_of_snake - 1
        training_stats['episode'].append(e + 1)
        training_stats['score'].append(score)
        training_stats['epsilon'].append(agent.epsilon)
        training_stats['total_reward'].append(episode_reward)
        recent_scores.append(score)
        recent_rewards.append(episode_reward)
        
        # 计算移动平均值
        avg_score = np.mean(recent_scores) if recent_scores else 0
        avg_loss = np.mean(recent_losses) if recent_losses else 0
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        training_stats['avg_score'].append(avg_score)
        training_stats['avg_reward'].append(avg_reward)
        
        # 打印每轮总结信息
        print(f"\n[EPISODE {e + 1}/{EPISODES}] Score: {score:3d}, Avg Score: {avg_score:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.6f}, Total Reward: {episode_reward:.2f}")
        print("=" * 70)
        
        # # 更新可视化图表
        # if show_visualization:
        #     update_plot(None)  # 手动更新图表
        #     plt.draw()  # 绘制最新数据
        #     plt.pause(0.1)  # 暂停以更新绘图
        
        # 每100轮保存一次模型
        if (e + 1) % 100 == 0:
            agent.save_model(f"snake_dqn_model_{e + 1}.pth")
            print(f"Model saved at episode {e + 1}")
    
    # 训练结束后保存最终模型
    agent.save_model(f"snake_dqn_model_final.pth")
    print(f"Final model saved")
    
    # 保持可视化窗口打开
    if show_visualization:
        plt.ioff()
        plt.show()


def play_mode(model_file):
    """加载训练好的模型并自动玩游戏"""
    game = GameCore()
    agent = DQNAgent(42, 4)
    
    # 加载模型
    try:
        agent.load_model(model_file)
        print(f"Model loaded from {model_file}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    state = game.reset()
    done = False
    
    while not done:
        # 使用模型选择动作（无探索）
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = agent.model(state_tensor)
        action = np.argmax(act_values.detach().numpy())
        
        # 执行动作
        next_state, reward, done = game.step(action)
        
        # 更新状态
        state = next_state
        
        # 渲染游戏
        game.render()
        
        # 控制游戏速度
        game.tick()
    
    print(f"Game over! Score: {game.Length_of_snake - 1}")


def manual_mode():
    """手动玩游戏模式"""
    game = GameCore()
    game.reset()
    
    # 初始化当前动作
    current_action = game.current_direction
    
    while not game.game_over:
        # 处理事件
        action = game.handle_events()
        
        if action == "quit":
            game.game_over = True
        elif action == "restart":
            game.reset()
            current_action = game.current_direction
        elif action is not None:
            # 更新当前动作
            current_action = action
        
        # 持续移动，无论是否有新的按键输入
        next_state, reward, done = game.step(current_action)
        print(f"Action: {current_action}, Reward: {reward}, Done: {done}")
        
        if done:
            # 游戏结束处理
            game.dis.fill(blue)
            game.message("You loss! Press Q-Quit or C-Play Again", red)
            game.Your_score()
            pygame.display.update()
            
            # 等待用户输入
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            game.game_over = True
                            waiting = False
                        elif event.key == pygame.K_c:
                            game.reset()
                            current_action = game.current_direction
                            waiting = False
        
        # 渲染游戏
        game.render()
        
        # 控制游戏速度
        game.tick(14)


def main():
    """主函数，解析命令行参数并启动相应模式"""
    parser = argparse.ArgumentParser(description='Snake Game DQN AI')
    
    # 模式选择参数
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='训练模式')
    group.add_argument('--play', action='store_true', help='自动玩模式')
    group.add_argument('--manual', action='store_true', help='手动玩模式')
    
    # 模型文件参数（仅在play模式下使用）
    parser.add_argument('--model', type=str, default='snake_dqn_model_final.pth', help='训练好的模型文件路径')
    
    args = parser.parse_args()
    
    print("=== Snake Game DQN AI ===")
    
    if args.play:
        # 自动玩模式
        print(f"启动自动玩模式，使用模型: {args.model}...")
        play_mode(args.model)
    elif args.manual:
        # 手动玩模式
        print("启动手动玩游戏模式...")
        manual_mode()
    else:
        # 默认模式：训练模式
        print("启动训练模式...")
        train_mode()


# 直接运行训练模式进行测试
if __name__ == "__main__":
    main()
    # 为了测试，直接运行训练模式
    # print("=== Snake Game DQN AI ===")
    # print("直接启动训练模式进行测试...")
    # train_mode(show_visualization=True)
    