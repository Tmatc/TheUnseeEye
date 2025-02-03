import pandas as pd
import numpy as np
from .base_result_handler import BaseResultHandler


class BasicResultHandler(BaseResultHandler):
    def handle_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        简单的后处理：裁剪负值预测，保留原始气象数据和预测结果。

        :param df: 包含预测结果的DataFrame
        :return: 后处理后的DataFrame
        """
        df = df.copy()
        if 'prediction' in df.columns:
            # 裁剪负值
            df['prediction'] = np.maximum(df['prediction'], 0)
        return df
