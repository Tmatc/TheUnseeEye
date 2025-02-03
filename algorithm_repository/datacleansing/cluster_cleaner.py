import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from .base_data_cleaner import BaseDataCleaner


class ClusteringDataCleaner(BaseDataCleaner):
    """
    适用于风电场历史数据，通过聚类算法识别并移除异常点和噪声数据。
    支持:
      1) DBSCAN
      2) KMeans
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'real_power'):
        """
        初始化数据清洗器。

        :param df: 原始DataFrame
        :param feature_cols: 用于聚类的特征列
        :param target_col: 目标列名(如 'real_power')
        """
        # 参数检查
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是一个pandas DataFrame。")

        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"特征列中缺少: {missing_features}")

        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在。")

        self.original_df = df.copy()
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.cleaned_df = None
        self.kmeans_model = None  # 添加KMeans模型属性
        self.dbscan_model = None  # 添加DBSCAN模型属性

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        形式实现，仅返回当前DataFrame。
        实际清洗需调用特定方法。
        """
        return self.df

    def handle_missing_values(self, strategy: str = 'mean'):
        """
        处理特征列中的缺失值。

        :param strategy: 填补策略，支持 'mean', 'median', 'mode'
        """
        valid_strategies = ['mean', 'median', 'mode']
        if strategy not in valid_strategies:
            raise ValueError(f"不支持的缺失值填补策略: {strategy}")

        if strategy == 'mean':
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(self.df[self.feature_cols].mean())
        elif strategy == 'median':
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(self.df[self.feature_cols].median())
        elif strategy == 'mode':
            mode_values = self.df[self.feature_cols].mode().iloc[0]
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(mode_values)

    def standardize_features(self):
        """
        对特征列进行标准化。
        """
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])

    def apply_dbscan(self, eps=0.5, min_samples=5):
        """
        应用DBSCAN聚类算法。

        :param eps: DBSCAN的eps参数
        :param min_samples: DBSCAN的min_samples参数
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(self.df[self.feature_cols])
        self.df['cluster'] = self.cluster_labels
        self.dbscan_model = dbscan  # 保存模型

    def identify_outliers_dbscan(self):
        """
        移除 DBSCAN 聚类结果中的噪声点 (cluster = -1)。
        """
        if 'cluster' not in self.df.columns:
            raise ValueError("请先调用 apply_dbscan。")

        self.cleaned_df = self.df[self.df['cluster'] != -1].copy()
        self.cleaned_df.drop(columns=['cluster'], inplace=True)

    def apply_kmeans(self, n_clusters=4):
        """
        应用KMeans聚类算法。

        :param n_clusters: KMeans的簇数
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.df[self.feature_cols])
        self.df['cluster'] = self.cluster_labels
        self.kmeans_model = kmeans  # 保存模型

    def identify_outliers_kmeans(self, threshold_std=2.0):
        """
        通过距离阈值移除KMeans聚类结果中的异常点。

        :param threshold_std: 阈值倍数，默认为2.0
        """
        if 'cluster' not in self.df.columns:
            raise ValueError("请先调用 apply_kmeans。")

        # 使用已拟合的KMeans模型
        if self.kmeans_model is None:
            raise ValueError("KMeans模型未拟合，请先调用 apply_kmeans。")

        # 计算每个样本到各簇中心的距离并取最小值
        distances = self.kmeans_model.transform(self.df[self.feature_cols])
        min_distances = np.min(distances, axis=1)
        self.df['distance'] = min_distances

        # 计算阈值
        threshold = self.df['distance'].mean() + threshold_std * self.df['distance'].std()

        # 标记并移除异常
        self.df['is_outlier'] = self.df['distance'] > threshold
        self.cleaned_df = self.df[~self.df['is_outlier']].copy()
        self.cleaned_df.drop(columns=['cluster', 'distance', 'is_outlier'], inplace=True)

    def get_cleaned_data(self) -> pd.DataFrame:
        """
        返回清洗后的结果。

        :return: 清洗后的DataFrame
        """
        if self.cleaned_df is None:
            raise ValueError("尚未执行异常识别或清洗操作。")
        return self.cleaned_df

    def reset(self):
        """
        重置为原始数据。
        """
        self.df = self.original_df.copy()
        self.cluster_labels = None
        self.cleaned_df = None
        self.kmeans_model = None
        self.dbscan_model = None
