from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gym

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        自定义特征提取器
        :param observation_space: (gym.spaces.Box) Observation space of the environment
        :param features_dim: (int) Number of features extracted by the network
        """
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.seq_len = 20  # 时间序列长度
        self.input_dim = observation_space.shape[1] - 1  # 除去仓位的维度
        self.hidden_dim = 128  # 隐藏层维度

        # 时间序列卷积层
        self.conv1d = nn.Conv1d(
            in_channels=self.seq_len, 
            out_channels=self.hidden_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        # 时间序列 LSTM 层
        self.lstm = nn.LSTM(
            input_size=self.input_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=2, 
            batch_first=True
        )

        # 当前仓位处理
        self.position_fc = nn.Linear(1, self.hidden_dim)

        # 特征融合后的全连接层
        self.fc = nn.Linear(self.hidden_dim * 2, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param observations: 输入数据，形状 (batch_size, observation_dim)
        """
        # 分割时间序列和仓位
        x_seq = observations[:, :-1].view(-1, self.seq_len, self.input_dim)  # 时间序列部分
        position = observations[:, -1:]  # 当前仓位部分

        # 时间序列特征提取
        x_seq = x_seq.permute(0, 2, 1)  # (batch_size, input_dim, seq_len) for Conv1D
        conv_out = torch.relu(self.conv1d(x_seq))  # (batch_size, hidden_dim, seq_len)
        conv_out = conv_out.permute(0, 2, 1)  # (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(conv_out)  # (batch_size, seq_len, hidden_dim)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步 (batch_size, hidden_dim)

        # 当前仓位特征提取
        pos_out = torch.relu(self.position_fc(position))  # (batch_size, hidden_dim)

        # 特征融合
        combined = torch.cat((lstm_out, pos_out), dim=1)  # (batch_size, hidden_dim * 2)
        features = torch.relu(self.fc(combined))  # (batch_size, features_dim)

        return features
