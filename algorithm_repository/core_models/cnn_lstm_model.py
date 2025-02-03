import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from .base_model import BaseModel


class CnnLstmNet(nn.Module):
    def __init__(self, window_size=24, feature_num=5, conv_channels=32, lstm_hidden=64):
        super(CnnLstmNet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=feature_num,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1  # 保持序列长度
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        """
        前向传播过程。

        :param x: 输入张量，形状为 (batch_size, seq_length, feature_num)
        :return: 输出张量，形状为 (batch_size, 1)
        """
        # 转换维度以匹配Conv1d的输入要求
        x = x.permute(0, 2, 1)  # 从 (batch, seq, features) 转为 (batch, features, seq)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # 转回 (batch, seq', features)
        output, (h_n, c_n) = self.lstm(x)
        x = output[:, -1, :]  # 取LSTM的最后一个时间步的输出
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CnnLstmModel(BaseModel):
    def __init__(self, window_size=24, feature_num=5, model_path='windpower_cnn_lstm_model.pt'):
        """
        初始化CNN+LSTM模型。

        :param window_size: 滑窗大小
        :param feature_num: 特征数量
        :param model_path: 模型保存路径
        """
        self.window_size = window_size
        self.feature_num = feature_num
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def build_model(self):
        """构建CNN+LSTM模型并初始化优化器。"""
        self.net = CnnLstmNet(
            window_size=self.window_size,
            feature_num=self.feature_num
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        print("[INFO] 模型架构已构建。")

    def train_model(self, X: np.ndarray, Y: np.ndarray):
        """
        使用训练数据训练模型。

        :param X: 特征数组，形状为 (num_samples, window_size, feature_num)
        :param Y: 目标数组，形状为 (num_samples,)
        """
        if self.net is None:
            self.build_model()

        # 转换为张量并移动到设备
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32).to(self.device)

        self.net.train()
        self.optimizer.zero_grad()
        preds = self.net(X_t).squeeze(-1)
        loss = self.criterion(preds, Y_t)
        loss.backward()
        self.optimizer.step()

        print(f"[INFO] 训练完成 - 损失: {loss.item():.4f}")

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """
        使用模型进行预测，并将结果添加到DataFrame中。

        :param X: 特征数组，形状为 (num_samples, window_size, feature_num)
        :return: 包含预测结果的DataFrame
        """
        if self.net is None:
            raise ValueError("模型尚未构建或加载。")

        # 转换为张量并移动到设备
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.net.eval()
        with torch.no_grad():
            preds = self.net(X_t).squeeze(-1).cpu().numpy()

        df = pd.DataFrame({'prediction': preds})
        return df

    def save_model(self):
        """保存模型参数到指定路径。"""
        if self.net is not None:
            torch.save(self.net.state_dict(), self.model_path)
            print(f"[INFO] 模型已保存到 {self.model_path}")
        else:
            print("[WARN] 模型为空，无法保存。")

    def load_model(self):
        """从指定路径加载模型参数。"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")

        if self.net is None:
            self.build_model()
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net.to(self.device)
        print(f"[INFO] 已加载模型参数: {self.model_path}")
