import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")


class TimeSeriesProcessor:
    def __init__(self, time_column, value_columns, time_interval='15T', missing_threshold=0, 
                 interpolation_method='linear'):
        """        
        time_column: 时间列的列名。
        value_columns: 需要处理的值列列表。
        time_interval: 时间间隔（默认15分钟）。
        missing_threshold: 连续缺失值的阈值，超过此值不进行插值。
        interpolation_method: 插值方法（支持 'linear', 'polynomial', 'spline', 'nearest', 'ffill', 'bfill'）。
        """
        self.time_column = time_column
        self.value_columns = value_columns
        self.time_interval = time_interval
        self.missing_threshold = missing_threshold
        self.interpolation_method = interpolation_method

    def process_data(self, df):
        """
        df: 输入的原始数据 DataFrame。
        
        返回:
        results: 包含以下内容的字典：
          - missing_times: 缺失的时间点列表。
          - filled_data: 插补的数据 DataFrame。
          - final_data: 最终的完整数据或最大连续子集 DataFrame。
        """
        # 确保时间列是 datetime 类型
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        # 创建完整的时间范围
        full_time_range = pd.date_range(start=df[self.time_column].min(), end=df[self.time_column].max(), freq=self.time_interval)
        
        # 找出缺失的时间点
        missing_times = full_time_range.difference(df[self.time_column])
        
        # 如果没有缺失的时间点，直接返回原始数据
        if missing_times.empty:
            return {
                'missing_times': [],
                'filled_data': pd.DataFrame(),
                'final_data': df
            }
        
        # 插补逻辑和最大子集选择
        filled_data, final_data = self._handle_data_segments(df, full_time_range)
        
        return {
            'missing_times': missing_times.tolist(),
            'filled_data': filled_data,
            'final_data': final_data
        }

    def _interpolate(self, x, y, x_new, method):
        """
        根据指定方法对数据进行插值。
        x: 原始时间点数组（数值型时间戳）。
        y: 原始数据值数组。
        x_new: 需要插值的新时间点数组。
        method: 插值方法。
        
        返回:
        插值后的新数据数组。
        """
        if method == 'linear':
            return np.interp(x_new, x, y)
        elif method == 'polynomial':
            poly_fit = np.polyfit(x, y, deg=min(3, len(x) - 1))
            poly = np.poly1d(poly_fit)
            return poly(x_new)
        elif method == 'spline':
            spline_fit = interp1d(x, y, kind='cubic', fill_value='extrapolate')
            return spline_fit(x_new)
        elif method == 'nearest':
            nearest_fit = interp1d(x, y, kind='nearest', fill_value='extrapolate')
            return nearest_fit(x_new)
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

    def _handle_data_segments(self, df, full_time_range):
        """
        根据缺失值的连续数量进行插补或提取最大连续子集。
        df: 原始数据。
        full_time_range: 完整的时间范围。
        
        返回:
        filled_data: 插补的数据 DataFrame。
        final_data: 最终的完整数据。
        """
        # 将原始数据重新索引，插入缺失的时间点
        df_full = df.set_index(self.time_column).reindex(full_time_range, method=None)
        
        # 标记缺失值的连续区间
        df_full['is_missing'] = df_full[self.value_columns].isnull().all(axis=1)
        df_full['missing_group'] = (df_full['is_missing'] != df_full['is_missing'].shift()).cumsum()
        
        # 获取连续缺失区间的长度
        missing_lengths = df_full[df_full['is_missing']].groupby('missing_group').size()
        
        # 检查是否有超过阈值的缺失段
        if any(length > self.missing_threshold for length in missing_lengths):
            # 找出最大连续子集
            df_full['valid'] = ~df_full['is_missing']
            df_full['valid_group'] = (df_full['valid'] != df_full['valid'].shift()).cumsum()
            valid_lengths = df_full[df_full['valid']].groupby('valid_group').size()
            max_valid_group = valid_lengths.idxmax()
            final_data = df_full[df_full['valid_group'] == max_valid_group].drop(columns=['is_missing', 'missing_group', 'valid', 'valid_group'])
            return pd.DataFrame(), final_data.reset_index().rename(columns={'index': self.time_column})
        
        # 处理插补逻辑
        filled_segments = []
        for group, length in missing_lengths.items():
            if length <= self.missing_threshold:
                # 对可以插补的部分进行插补
                segment = df_full[df_full['missing_group'] == group]
                for col in self.value_columns:
                    non_missing = df_full[~df_full[col].isnull()]
                    if len(non_missing) > 1:
                        x = non_missing.index.view('int64') / 10**9  # 转换为秒数
                        y = non_missing[col].values
                        x_new = segment.index.view('int64') / 10**9  # 转换为秒数
                        segment[col] = self._interpolate(x, y, x_new, self.interpolation_method)
                filled_segments.append(segment)
        
        # 合并插值后的数据段
        filled_data = pd.concat(filled_segments) if filled_segments else pd.DataFrame()
        
        # 更新插值后的完整数据
        if not filled_data.empty:
            df_full.update(filled_data)
        
        # 删除临时列
        df_full.drop(columns=['is_missing', 'missing_group'], inplace=True)
        
        # 保证最终数据完整，包括插值和原始数据
        final_data = df_full.reset_index().rename(columns={'index': self.time_column})
        
        return filled_data.reset_index().rename(columns={'index': self.time_column}), final_data


if __name__ == "__main__":
    # 示例数据
    data = {
        'time': [
            '2025-01-01 00:00:00', '2025-01-01 00:15:00', '2025-01-01 00:30:00',
            '2025-01-01 00:45:00', '2025-01-01 01:00:00', '2025-01-01 02:00:00',
            '2025-01-01 02:15:00', '2025-01-01 02:30:00', '2025-01-01 03:30:00',
            '2025-01-01 03:45:00', '2025-01-01 04:00:00', '2025-01-01 04:15:00',
            '2025-01-01 04:30:00', '2025-01-01 04:45:00', '2025-01-01 05:45:00',
            '2025-01-01 06:00:00', '2025-01-01 06:15:00', '2025-01-01 06:30:00',
            '2025-01-01 06:45:00', '2025-01-01 07:00:00'
        ],
        'value': [
            1, 2, 3, 4, 5, 11, 12, 13, 19, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35
        ]
    }
    df = pd.DataFrame(data)


    # 创建处理器
    processor = TimeSeriesProcessor(time_column='time', value_columns=['value'], time_interval='15T', 
                                    missing_threshold=0, interpolation_method='linear')

    # 处理数据
    results = processor.process_data(df)

    # 输出结果
    print("缺失的时间点：", results['missing_times'])
    print("\n插补的数据：")
    print(results['filled_data'])
    print("\n最终的数据：")
    print(results['final_data'])
