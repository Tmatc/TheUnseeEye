from abc import ABC, abstractmethod
import pandas as pd


class BaseResultHandler(ABC):
    @abstractmethod
    def handle_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        接收包含预测结果的DataFrame，并进行后处理。
        """
        pass
