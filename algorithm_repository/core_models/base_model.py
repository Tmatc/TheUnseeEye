# algorithm_repository/core_models/base_model.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def build_model(self):
        """构建模型架构。"""
        pass

    @abstractmethod
    def train_model(self, X: np.ndarray, Y: np.ndarray):
        """使用训练数据训练模型。"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """使用模型进行预测。"""
        pass

    @abstractmethod
    def save_model(self):
        """保存模型参数。"""
        pass

    @abstractmethod
    def load_model(self):
        """加载模型参数。"""
        pass
