from abc import ABC, abstractmethod
import pandas as pd


class BaseDataCleaner(ABC):
    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        接收一个DataFrame进行清洗，并返回清洗后的DataFrame。
        """
        pass
