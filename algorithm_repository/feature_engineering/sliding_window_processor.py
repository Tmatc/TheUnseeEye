import pandas as pd
import numpy as np
from .base_feature_processor import BaseFeatureProcessor


class SlidingWindowProcessor(BaseFeatureProcessor):
    def __init__(self, window_size=24, feature_cols=None, target_col='real_power'):
        self.window_size = window_size
        if feature_cols is None:
            self.feature_cols = ['wind_speed', 'wind_direction', 'humidity', 'temperature', 'pressure']
        else:
            self.feature_cols = feature_cols
        self.target_col = target_col

    def process_features(self, df: pd.DataFrame, is_train=True):
        """
        使用滑动窗口生成特征和目标数组。

        :param df: 清洗后的DataFrame
        :param is_train: 是否为训练模式
        :return: (X, Y) 元组，其中X为特征数组，Y为目标数组（仅训练模式）
        """
        # 校验
        for col in self.feature_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列：{col}")
        if is_train and self.target_col not in df.columns:
            raise ValueError(f"缺少目标列：{self.target_col}")

        feature_data = df[self.feature_cols].values
        if is_train:
            target_data = df[self.target_col].values

        X_list = []
        Y_list = [] if is_train else None
        for i in range(len(df) - self.window_size):
            X_list.append(feature_data[i : i + self.window_size])
            if is_train:
                Y_list.append(target_data[i + self.window_size])

        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32) if is_train else None

        if is_train:
            return X, Y
        else:
            return X
