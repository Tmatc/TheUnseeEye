import numpy as np
import pandas as pd
from core.resultservice.visualization.err_rate_vs_self import generate_visualization

class NewVsOldErrorRateCurve:
    def __init__(self, theme_name, 
                 time_data:list, 
                 oldpre:list, 
                 newpre:list, 
                 realp:list, 
                 olderrrate:list, 
                 newerrrate:list, warning_threshold=0.2, 
                 legend_names=["old Error Rate", "new Error Rate", "old Predictions", 
                               "new Predictions", "Actual Power"]
                ):
        '''
        theme_name: str, 主题名称
        time_data: list, 时间数据
        oldpre: list, 旧模型预测值
        newpre: list, 新模型预测值
        realp: list, 实际值
        olderrrate: list, 旧模型误差率
        newerrrate: list, 新模型误差率
        warning_threshold: float, 警戒线
        legend_names: list, 图例名称
        '''
        time_data = time_data  

        # 模拟模型1和模型2的预测值
        model1_predictions = oldpre  
        model2_predictions = newpre 
        actual_values = realp # 实际值

        # 计算误差率
        model1_errors = olderrrate   
        model2_errors = newerrrate  

        warning_threshold = warning_threshold 
        theme_name = theme_name  # 传入的主题名称
        legend_names = legend_names
        # 调用生成图表的函数，并指定图表宽高
        generate_visualization(model1_predictions, model2_predictions, actual_values, 
                            model1_errors, model2_errors, warning_threshold, theme_name, 
                            legend_names, time_data, width="1600px", height="450px")
