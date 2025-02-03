from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from ..core_models.base_model import BaseModel


class BaseTrainer(ABC):
    @abstractmethod
    def train_model(self, model: BaseModel, X: np.ndarray, Y: np.ndarray):
        """
        接收一个模型实例 + 特征工程/清洗后的数据，并执行训练。
        """
        pass
