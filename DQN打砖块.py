import gym
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 1. 优先级经验回放池（自定义实现）
class SumTree:
    """用于高效存储和采样优先级的SumTree结构"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 树的总节点数
        self.data = np.zeros(capacity, dtype=object)  # 存储数据
        self.write = 0  # 当前写入位置

    def _propagate(self, idx, change):
        """更新父节点的优先级值"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """根据优先级值找到对应的叶节点"""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """获取总优先级值"""
        return self.tree[0]

    def add(self, p, data):
        """添加新数据及其优先级"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity

    def update(self, idx, p):
        """更新指定节点的优先级"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """根据优先级值采样数据"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta_start  # 重要性采样指数
        self.beta_increment_per_step = (1.0 - beta_start) / beta_frames  # beta随时间增加
        self.epsilon = 1e-6  # 避免优先级为0

    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        """根据优先级采样批次数据"""
        batch_idx, batch_data = [], []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment_per_step)  # 增加beta值

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch_idx.append(idx)
            batch_data.append(data)

        # 计算重要性采样权重
        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # 归一化权重

        states, actions, rewards, next_states, dones = zip(*batch_data)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            batch_idx,
            is_weights
        )

    def update_priorities(self, idxes, priorities):
        """更新样本的优先级"""
        for idx, p in zip(idxes, priorities):
            p = (p + self.epsilon) ** self.alpha  # 应用alpha指数
            self.tree.update(idx, p)

    def __len__(self):
        """返回缓冲区当前大小"""
        count = 0
        for data in self.tree.data:
            if data is not None:
                count += 1
        return count


# 2. Q网络结构
class DQNNet(nn.Module):
    def __init__(self, input_channels=4, action_dim=4):
        super(DQNNet, self).__init__()

        # 特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.res_block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )
        self.res_relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.res_relu2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 注意力机制
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128//16, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128//16, 128, kernel_size=1),
            nn.Sigmoid()
        )

        # Dueling DQN
        self.value_stream = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.res_block1(x)
        x += residual
        x = self.res_relu1(x)

        x = self.conv2(x)
        residual = x
        x = self.res_block2(x)
        x += residual
        x = self.res_relu2(x)

        x = self.conv3(x)

        # 注意力加权
        spatial_weights = self.spatial_attn(x)
        channel_weights = self.channel_attn(x)
        x = x * spatial_weights * channel_weights

        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


# 3. DQN主类
class DQNAgent:
    def __init__(self, env, input_shape=(84, 84), stack_frames=4, replay_capacity=200000):
        self.env = env
        self.action_dim = env.action_space.n
        self.stack_frames = stack_frames
        self.input_shape = input_shape

        self.eval_net = DQNNet(input_channels=stack_frames, action_dim=self.action_dim).to(device)
        self.target_net = DQNNet(input_channels=stack_frames, action_dim=self.action_dim).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=0.0000625)

        # 优先级经验回放
        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity)
        self.batch_size = 32
        self.replay_start_size = 20000

        self.gamma = 0.99
        self.target_update_freq = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = (1.0 - 0.01) / 200000
        self.learn_step = 0
        self.max_grad_norm = 5

        self.start_learning = False

        # 奖励机制参数
        self.last_ball_pos = None
        self.ball_threshold = 180
        self.paddle_threshold = 120
        self.bottom_threshold = 70
        self.penalty_value = -3.0
        self.break_brick_reward = 2.0

    def preprocess_observation(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.input_shape, interpolation=cv2.INTER_AREA)
        self.last_ball_pos = self.detect_ball_position(resized)
        return resized.astype(np.float32) / 255.0

    def stack_observation(self, state, new_obs):
        processed_obs = self.preprocess_observation(new_obs)
        if state is None:
            return np.stack([processed_obs] * self.stack_frames, axis=0)
        else:
            return np.concatenate([state[1:], [processed_obs]], axis=0)

    def detect_ball_position(self, frame):
        ball_mask = (frame > self.ball_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3 < area < 30:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        return None

    def detect_paddle_position(self, frame):
        bottom_region = frame[self.bottom_threshold:, :]
        paddle_mask = (bottom_region < self.paddle_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(paddle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                return (int(M["m10"]/M["m00"]), self.bottom_threshold + int(M["m01"]/M["m00"]))
        return None

    def calculate_penalty(self, state, done):
        if done and self.last_ball_pos:
            ball_x, ball_y = self.last_ball_pos
            if ball_y > self.bottom_threshold:
                paddle_pos = self.detect_paddle_position(state[-1])
                if paddle_pos:
                    paddle_x, _ = paddle_pos
                    if abs(paddle_x - ball_x) > 12:
                        return self.penalty_value
        return 0.0

    def select_action(self, state):
        if not self.start_learning:
            return np.random.randint(self.action_dim)

        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.eval_net(state_tensor)).item()

    def learn(self):
        if len(self.replay_buffer) < self.replay_start_size:
            return None

        if not self.start_learning:
            self.start_learning = True
            print("开始学习！")

        # 优先级采样
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)


        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        weights = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

        # Double DQN
        q_eval = self.eval_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.eval_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * q_next

        # 计算误差并更新优先级
        errors = torch.abs(q_eval - q_target).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, errors)

        # 带权重的损失
        loss = (weights * F.smooth_l1_loss(q_eval, q_target, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 更新目标网络
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        return loss.item()

    def save_model(self, path):
        torch.save({
            'eval_net': self.eval_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learn_step': self.learn_step
        }, path)
        print(f"模型保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.eval_net.load_state_dict(checkpoint['eval_net'])
        self.target_net.load_state_dict(checkpoint['eval_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_step = checkpoint['learn_step']
        print(f"模型加载自: {path}")


# 4. 训练主函数
def train_breakout():
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    writer = SummaryWriter(log_dir='./breakout_logs')

    agent = DQNAgent(env)
    load_model = False
    if load_model:
        agent.load_model('./best.pth')

    total_episodes = 1000
    max_steps = 10000
    best_score = 0

    for episode in range(total_episodes):
        obs, _ = env.reset()
        state = agent.stack_observation(None, obs)
        episode_reward = 0.0
        episode_loss = []

        for step in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 强化奖励
            if reward > 0:
                reward += agent.break_brick_reward
            penalty = agent.calculate_penalty(state, done)
            reward += penalty

            next_state = agent.stack_observation(state, next_obs)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            episode_reward += reward
            if done:
                break

        # 记录日志
        if episode % 5 == 0:
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            print(f"回合 {episode} | 奖励: {episode_reward:.0f} | 损失: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | 样本: {len(agent.replay_buffer)}")
            writer.add_scalar('Reward', episode_reward, episode)
            writer.add_scalar('Loss', avg_loss, episode)

        # 保存最佳模型
        if episode_reward > best_score and agent.start_learning:
            best_score = episode_reward
            agent.save_model(f'./best_{episode}_{best_score:.0f}.pth')
            print(f"最佳奖励更新: {best_score}")

    writer.close()
    env.close()



if __name__ == "__main__":
   train_breakout()
 #trained_model_path = "./best_593_60.pth"
 #test_breakout(trained_model_path)
 #test_breakout('./breakout_best_model.pth')