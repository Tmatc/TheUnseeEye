from .base_evaluator import BaseEvaluator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class BasicEvaluator(BaseEvaluator):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        计算MSE和MAE指标。

        :param y_true: 真实值
        :param y_pred: 预测值
        :return: 包含评估指标的字典
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {
            'MSE': mse,
            'MAE': mae
        }
