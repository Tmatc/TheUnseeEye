from abc import ABC, abstractmethod
import numpy as np


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        接收真实值和预测值，返回评估指标。
        """
        pass
