# algorithm_repository/datacleansing/power_cleaner.py

import pandas as pd
import numpy as np

def clean_power_by_theory(
    raw_data: pd.DataFrame,
    theoretical_power: pd.DataFrame,
    raw_wind_speed_col: str,
    error_margin: float = 0.5
) -> pd.DataFrame:
    """
    根据给定的理论功率曲线，对原始数据中实测功率进行过滤，
    移除超出理论功率 ± (error_margin倍数) 的样本。

    :param raw_data: 原始数据，包含 wind_speed 列和 real_power 列
    :param theoretical_power: 理论功率表，包含列: [wind_speed, power]
    :param raw_wind_speed_col: 原始数据中的风速列名
    :param error_margin: 容忍的偏差倍数(默认0.5 => ±50%)
    :return: 清洗后的DataFrame
    """
    required_raw_columns = {raw_wind_speed_col, 'real_power'}
    required_theoretical = {'wind_speed', 'power'}

    if not required_raw_columns.issubset(raw_data.columns):
        raise ValueError(f"raw_data必须包含列: {required_raw_columns}")

    if not required_theoretical.issubset(theoretical_power.columns):
        raise ValueError(f"theoretical_power必须包含列: {required_theoretical}")

    # 为了安全起见，复制一份
    raw_data = raw_data.copy()
    theoretical_power = theoretical_power.copy()

    # 排序并插值
    theoretical_power.sort_values('wind_speed', inplace=True)
    theoretical_power.reset_index(drop=True, inplace=True)

    # 使用 np.interp 进行插值
    raw_data['theoretical_power'] = np.interp(
        raw_data[raw_wind_speed_col],
        theoretical_power['wind_speed'],
        theoretical_power['power']
    )

    # 计算允许范围
    raw_data['min_power'] = raw_data['theoretical_power'] * (1 - error_margin)
    raw_data['max_power'] = raw_data['theoretical_power'] * (1 + error_margin)

    # 过滤
    cleaned_data = raw_data[
        (raw_data['real_power'] >= raw_data['min_power']) &
        (raw_data['real_power'] <= raw_data['max_power'])
    ].copy()

    # 删除临时列
    cleaned_data.drop(columns=['theoretical_power', 'min_power', 'max_power'], inplace=True)

    return cleaned_data
