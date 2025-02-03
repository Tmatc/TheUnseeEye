import numpy as np
import pandas as pd
from core.resultservice.errorrate_curve import NewVsOldErrorRateCurve


if __name__ == "__main__":

    theme_name = "DRF3001 pred"  # 传入的主题名称

    # 生成3000个测试数据点
    time_data = pd.date_range(start="2023-01-01", periods=30000, freq="15T").strftime('%Y-%m-%d %H:%M:%S').tolist()  # 3000个数据点

    # 模拟模型1和模型2的预测值
    model1_predictions = np.random.uniform(10, 15, 30000).tolist()  # 模型1预测值
    model2_predictions = np.random.uniform(10, 15, 30000).tolist()  # 模型2预测值
    actual_values = np.random.uniform(10, 15, 30000).tolist()  # 实际值

    # 计算误差率
    model1_errors = np.abs(np.array(model1_predictions) - np.array(actual_values)) / np.array(actual_values)
    model2_errors = np.abs(np.array(model2_predictions) - np.array(actual_values)) / np.array(actual_values)

    warning_threshold = 0.2  # 假设误差率警戒线为15%
    legend_names = ["Model 1 Error Rate", "Model 2 Error Rate", 
                    "Model 1 Predictions", "Model 2 Predictions", "Actual Power"]
    NewVsOldErrorRateCurve(theme_name, time_data, model1_predictions, model2_predictions,actual_values,
                           model1_errors, model2_errors, warning_threshold)
    
