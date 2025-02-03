import pandas as pd
import numpy as np

# ========== 数据清洗 ==========
from algorithm_repository.datacleansing.cluster_cleaner import ClusteringDataCleaner
from algorithm_repository.datacleansing.power_cleaner import clean_power_by_theory

# ========== 特征工程 ==========
from algorithm_repository.feature_engineering.sliding_window_processor import SlidingWindowProcessor

# ========== 模型 (PyTorch CNN+LSTM) ==========
from algorithm_repository.core_models.cnn_lstm_model import CnnLstmModel

# ========== 训练逻辑 (带数据拆分、早停等) ==========
from algorithm_repository.training_logic.advanced_trainer import AdvancedTrainer

# ========== 评估 & 结果处理 ==========
from algorithm_repository.evaluation.basic_evaluator import BasicEvaluator
from algorithm_repository.resultlogic.basic_result_handler import BasicResultHandler

class CnnLstmWindPowerApp:
    """
    应用模型示例：

      - rename_columns: 重命名数据列
      - train:       训练模型
      - predict:     使用模型进行预测
      - evaluate:    评估模型性能

    统一采用列名:
      - 风速列: wind_speed
      - 风向列: wind_direction
      - 湿度列: humidity
      - 温度列: temperature
      - 气压列: pressure
      - 实测功率列: real_power
      - 时间列: dtime
    """

    def __init__(
        self,
        window_size=24,
        feature_num=5,  # wind_speed, wind_direction, humidity, temperature, pressure
        model_path='windpower_cnn_lstm_model.pt',
        batch_size=32,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        early_stop_patience: int = 5,
        save_best_model: bool = True
    ):
        """
        初始化应用。

        :param window_size: 滑窗大小
        :param feature_num: 特征数=5 (wind_speed, wind_direction, humidity, temperature, pressure)
        :param model_path: 模型保存/加载路径
        :param batch_size: 批大小
        :param learning_rate: 学习率
        :param epochs: 训练轮数
        :param early_stop_patience: 早停轮数
        :param save_best_model: 是否保存最佳模型
        """
        self.window_size = window_size
        self.feature_num = feature_num
        self.model_path = model_path

        # ========== 构建模型 ==========
        self.model = CnnLstmModel(
            window_size=self.window_size,
            feature_num=self.feature_num,
            model_path=self.model_path
        )

        # ========== 构建训练器 (带早停等) ==========
        self.trainer = AdvancedTrainer(
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            save_best_model=save_best_model
        )

        # ========== 评估 & 结果处理器 ==========
        self.evaluator = BasicEvaluator()
        self.result_handler = BasicResultHandler()

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        重命名原始数据的列，以匹配内部处理流程。

        :param df: 原始DataFrame
        :return: 重命名后的DataFrame
        """
        renamed_df = df.rename(columns={
            'speed70': 'wind_speed',
            'direction70': 'wind_direction',
            'humidity70': 'humidity',
            'temperature70': 'temperature',
            'pressure70': 'pressure'
            # 'real_power' 和 'dtime' 保持不变
        })
        return renamed_df

    def train(self, df: pd.DataFrame, df_theory: pd.DataFrame = None, cleaning_method='cluster'):
        """
        训练模型。

        :param df: 原始数据
        :param df_theory: 理论功率数据，若使用理论清洗则需要提供
        :param cleaning_method: 数据清洗方法，'cluster' 或 'theory'
        """
        # 1. 重命名列
        df_renamed = self.rename_columns(df)

        # 2. 数据清洗（选择聚类或理论清洗）
        if cleaning_method == 'cluster':
            cleaned_df = self.clean_data_with_cluster(
                df=df_renamed,
                feature_cols=['wind_speed', 'wind_direction', 'humidity', 'temperature', 'pressure'],
                method='dbscan',
                eps=1.0,
                min_samples=2
            )
        elif cleaning_method == 'theory':
            if df_theory is None:
                raise ValueError("理论功率数据缺失。")
            cleaned_df = self.clean_data_with_theory(
                raw_data=df_renamed,
                theoretical_data=df_theory,
                raw_wind_speed_col='wind_speed',
                error_margin=0.3
            )
        else:
            raise ValueError("cleaning_method必须是 'cluster' 或 'theory'")

        # 3. 特征工程
        X, Y = self.process_features(cleaned_df, is_train=True)

        # 4. 训练模型
        self.trainer.train_model(self.model, X, Y)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用模型进行预测。

        :param df: 需要预测的数据
        :return: 包含预测结果的DataFrame
        """
        # 1. 重命名列
        df_renamed = self.rename_columns(df)

        # 2. 数据清洗（不需要 real_power）
        df_no_power = df_renamed.copy()
        df_no_power['real_power'] = np.nan  # 填充NaN以满足ClusteringDataCleaner的要求

        # 使用聚类清洗
        cleaner = ClusteringDataCleaner(
            df=df_no_power,
            feature_cols=['wind_speed', 'wind_direction', 'humidity', 'temperature', 'pressure'],
            target_col='real_power'
        )
        # 处理缺失值和标准化
        cleaner.handle_missing_values(strategy='mean')
        cleaner.standardize_features()

        # 应用DBSCAN
        cleaner.apply_dbscan(eps=1.0, min_samples=2)
        cleaner.identify_outliers_dbscan()
        cleaned_df = cleaner.get_cleaned_data()

        # 添加调试信息
        print(f"[DEBUG] 清洗后的数据点数量: {len(cleaned_df)}")

        # 3. 特征工程
        X = self.process_features(cleaned_df, is_train=False)
        print(f"[DEBUG] 预测模式 - 特征数组形状: {X.shape}")

        # 检查 X 是否为空
        if X.size == 0:
            raise ValueError("滑动窗口生成的特征数组为空，请检查输入数据量和滑动窗口大小。")

        # 4. 预测
        self.model.load_model()
        predictions_df = self.model.predict(X)
        print(f"[DEBUG] 预测结果: \n{predictions_df.head()}")

        # 5. 后处理
        final_df = self.result_handler.handle_result(predictions_df)

        # 6. 将预测结果与原始数据的时间对应
        window_size = self.window_size
        prediction_length = len(final_df['prediction'])
        if len(df) < prediction_length:
            raise ValueError("预测结果的长度超过了原始数据的长度。")
        time_series = df['dtime'].values[-prediction_length:]

        # 构建结果DataFrame
        result_df = pd.DataFrame({
            'dtime': time_series,
            'predicted_real_power': final_df['prediction']
        })

        return result_df

    def evaluate(self, df_with_pred: pd.DataFrame) -> dict:
        """
        评估模型性能。

        :param df_with_pred: 包含预测结果和真实值的DataFrame
        :return: 评估指标字典
        """
        if 'predicted_real_power' not in df_with_pred.columns:
            print("[WARN] 无法评估：缺少 'predicted_real_power' 列")
            return {}

        # 评估需要真实值，这里假设真实值在传入DataFrame中
        if 'real_power' not in df_with_pred.columns:
            print("[WARN] 无法评估：缺少 'real_power' 列")
            return {}

        preds = df_with_pred['predicted_real_power'].values
        y_true = df_with_pred['real_power'].values

        if len(y_true) != len(preds):
            print("[WARN] 评估失败：真实值和预测值长度不一致。")
            return {}

        metrics = self.evaluator.evaluate(y_true, preds)
        print("[INFO] 评估指标:", metrics)
        return metrics

    def clean_data_with_cluster(self, df: pd.DataFrame, feature_cols: list, method='dbscan', **kwargs) -> pd.DataFrame:
        """
        使用聚类清洗 (DBSCAN 或 KMeans)。

        :param df: 重命名后的DataFrame
        :param feature_cols: 用于聚类的特征列
        :param method: 聚类方法，'dbscan' 或 'kmeans'
        :param kwargs: 额外参数，如eps、min_samples等
        :return: 清洗后的DataFrame
        """
        cleaner = ClusteringDataCleaner(
            df=df,
            feature_cols=feature_cols,
            target_col='real_power'
        )
        # 缺失值处理 + 标准化
        cleaner.handle_missing_values(strategy='mean')
        cleaner.standardize_features()

        if method.lower() == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            cleaner.apply_dbscan(eps=eps, min_samples=min_samples)
            cleaner.identify_outliers_dbscan()
        elif method.lower() == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 4)
            cleaner.apply_kmeans(n_clusters=n_clusters)
            threshold_std = kwargs.get('threshold_std', 2.0)
            cleaner.identify_outliers_kmeans(threshold_std=threshold_std)
        else:
            raise ValueError("method必须是 'dbscan' 或 'kmeans'")

        return cleaner.get_cleaned_data()

    def clean_data_with_theory(
        self,
        raw_data: pd.DataFrame,
        theoretical_data: pd.DataFrame,
        raw_wind_speed_col='wind_speed',
        error_margin=0.5
    ) -> pd.DataFrame:
        """
        使用理论功率清洗。

        :param raw_data: 重命名后的原始数据
        :param theoretical_data: 理论功率数据
        :param raw_wind_speed_col: 原始数据中的风速列名
        :param error_margin: 容忍的偏差倍数
        :return: 清洗后的DataFrame
        """
        cleaned = clean_power_by_theory(
            raw_data,
            theoretical_data,
            raw_wind_speed_col=raw_wind_speed_col,
            error_margin=error_margin
        )
        print(f"[DEBUG] 使用理论功率清洗后数据点数量: {len(cleaned)}")
        return cleaned

    def process_features(self, df: pd.DataFrame, is_train=True):
        """
        进行特征工程，使用滑动窗口处理。

        :param df: 清洗后的DataFrame
        :param is_train: 是否为训练模式
        :return: (X, Y) 元组（训练模式）或 X 数组（预测模式）
        """
        processor = SlidingWindowProcessor(window_size=self.window_size)
        if is_train:
            X, Y = processor.process_features(df, is_train=is_train)
            print(f"[DEBUG] 训练模式 - X形状: {X.shape}, Y形状: {Y.shape}")
            return X, Y
        else:
            X = processor.process_features(df, is_train=is_train)
            print(f"[DEBUG] 预测模式 - X形状: {X.shape}")
            return X