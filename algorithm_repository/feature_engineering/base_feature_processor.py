from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureProcessor(ABC):
    @abstractmethod
    def process_features(self, df: pd.DataFrame, is_train=True):
        """
        接收一个DataFrame进行特征工程，并返回包含特征和目标的DataFrame或元组。
        """
        pass
