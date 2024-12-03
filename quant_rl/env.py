import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, data, window_size=20, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Observation space: 20天的股票数据和当前仓位
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size * data.shape[1] + 1,), dtype=np.float32
        )

        # Action space: 调整后的仓位比例 [0, 1]
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        # 处理种子
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 当前仓位
        self.relative_position = 0
        self.total_shares = 0
        self.start_balance = self.balance
        self.last_total_assets = self.balance
        self.done = False
        self.trades = []

        return self.get_observation(), {}

    def get_observation(self):
        # 提取前 window_size 天的数据
        obs_data = self.data.iloc[self.current_step - self.window_size : self.current_step].values.flatten()
        return np.append(obs_data, self.relative_position).astype(np.float32)  # 转换为 float32


    def step(self, action):
        if self.done:
            raise ValueError("Episode already done")

        # 将 action 映射到仓位调整比例
        target_position = float(action[0]) * self.balance / self.data.iloc[self.current_step]['close']
        delta_position = target_position - self.position

        # 执行买卖操作
        current_price = self.data.iloc[self.current_step]['close']
        if delta_position > 0:
            # 买入
            cost = delta_position * current_price
            if cost <= self.balance:
                self.balance -= cost
                self.position += delta_position
                self.trades.append(("buy", delta_position, current_price))
        elif delta_position < 0:
            # 卖出
            self.balance += abs(delta_position) * current_price
            self.position += delta_position
            self.trades.append(("sell", delta_position, current_price))

        # 移动到下一步
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True
        truncated = False  # 没有特定的截断逻辑

        # 计算收益率（Reward）
        total_assets = self.balance + self.position * current_price
        reward = (total_assets - self.last_total_assets) / self.last_total_assets
        self.last_total_assets = total_assets
        self.relative_position = self.position * current_price / total_assets

        return self.get_observation(), reward, self.done, truncated, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.2f}, relative_position: {self.relative_position:.2f}")
