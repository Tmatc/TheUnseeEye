import numpy as np
import re
from pyecharts import options as opts
from pyecharts.charts import Line, Page
import pandas as pd


'''
此模块用于生成两个图表：对比新旧两个模型的预测值与实际值，以及两个模型的误差率与实际值的关系。
'''
# 定义数据类
class PredictionData:
    def __init__(self, time_data, model1_predictions, model2_predictions, actual_values, model1_errors, model2_errors):
        self.time_data = time_data
        self.model1_predictions = model1_predictions
        self.model2_predictions = model2_predictions
        self.actual_values = actual_values
        self.model1_errors = model1_errors
        self.model2_errors = model2_errors

# 定义可视化类
class Visualizer:
    def __init__(self, prediction_data, warning_threshold, theme_name, legend_names, width="1200px", height="600px"):
        self.prediction_data = prediction_data
        self.warning_threshold = warning_threshold
        self.theme_name = theme_name
        self.legend_names = legend_names
        self.width = width  # 自定义图表宽度
        self.height = height  # 自定义图表高度

    def plot_error_rate(self):
        """
        绘制模型1和模型2的误差率与实际值的关系
        """
        line = Line(init_opts=opts.InitOpts(width=self.width, height=self.height))  # 设置宽度和高度
        line.add_xaxis(self.prediction_data.time_data)  # 使用时间轴数据
        
        # 格式化数据只显示两位小数
        model1_errors = [round(x, 2) for x in self.prediction_data.model1_errors]
        model2_errors = [round(x, 2) for x in self.prediction_data.model2_errors]
        
        # 设置纵坐标的范围，确保覆盖所有的误差率
        min_error = min(min(model1_errors), min(model2_errors))
        max_error = max(max(model1_errors), max(model2_errors))
        
        # 模型1误差率为绿色虚线，点为更大的实心圆点
        line.add_yaxis(self.legend_names[0], model1_errors, color="green", 
                       linestyle_opts=opts.LineStyleOpts(type_="dashed", width=1), 
                       symbol="circle", symbol_size=8, label_opts=opts.LabelOpts(is_show=False), 
                       is_smooth=False)  
        # 模型2误差率为橙色虚线，点为更大的实心圆点
        line.add_yaxis(self.legend_names[1], model2_errors, color="orange", 
                       linestyle_opts=opts.LineStyleOpts(type_="dashed", width=1), 
                       symbol="circle", symbol_size=8, label_opts=opts.LabelOpts(is_show=False), 
                       is_smooth=False)  
        
        # 绘制警戒线（红色水平线）
        line.add_yaxis(f"Error Threshold ({self.warning_threshold*100}%)", 
                       [self.warning_threshold] * len(self.prediction_data.time_data), color="red", 
                       linestyle_opts=opts.LineStyleOpts(type_="dashed", width=2), label_opts=opts.LabelOpts(is_show=False),
                       symbol="none", is_smooth=False)  # 红色虚线，表示警戒线

        line.set_global_opts(
            title_opts=opts.TitleOpts(title=self.theme_name),  # 使用传入的主题名称作为标题
            xaxis_opts=opts.AxisOpts(name="Time", type_="category", boundary_gap=False, axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name="Error Rate", min_=min_error - 0.01, max_=max_error + 0.01),  # 动态调整纵坐标范围，防止曲线裁剪
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="15%", pos_right="10%", orient="horizontal"),  # 图例放到右上方，横向排列
            datazoom_opts=opts.DataZoomOpts(is_show=True, type_="slider", range_start=0, range_end=96),  # 滑动条显示96个数据点
        )
        return line

    def plot_prediction_vs_actual(self):
        """
        绘制模型1、模型2与实际功率的对比
        """
        line = Line(init_opts=opts.InitOpts(width=self.width, height=self.height))  # 设置宽度和高度
        line.add_xaxis(self.prediction_data.time_data)  # 使用时间轴数据
        
        # 格式化数据只显示两位小数
        model1_predictions = [round(x, 2) for x in self.prediction_data.model1_predictions]
        model2_predictions = [round(x, 2) for x in self.prediction_data.model2_predictions]
        actual_values = [round(x, 2) for x in self.prediction_data.actual_values]
        
        # 预测值为绿色实线，点为实心圆点
        line.add_yaxis(self.legend_names[2], model1_predictions, color="green", 
                       linestyle_opts=opts.LineStyleOpts(type_="solid", width=2), 
                       symbol="circle", symbol_size=6, label_opts=opts.LabelOpts(is_show=False), 
                       is_smooth=True)  
        # 预测值为蓝色实线，点为实心圆点
        line.add_yaxis(self.legend_names[3], model2_predictions, color="blue", 
                       linestyle_opts=opts.LineStyleOpts(type_="solid", width=2), 
                       symbol="circle", symbol_size=6, label_opts=opts.LabelOpts(is_show=False), 
                       is_smooth=True)  
        # 实际值为橘黄色实线，点为实心圆点
        line.add_yaxis(self.legend_names[4], actual_values, color="orange", 
                       linestyle_opts=opts.LineStyleOpts(type_="solid", width=2), 
                       symbol="circle", symbol_size=6, label_opts=opts.LabelOpts(is_show=False), 
                       is_smooth=True)  

        line.set_global_opts(
            title_opts=opts.TitleOpts(title=self.theme_name),  # 使用传入的主题名称作为标题
            xaxis_opts=opts.AxisOpts(name="Time", type_="category", boundary_gap=False, axislabel_opts=opts.LabelOpts(rotate=45)),
            yaxis_opts=opts.AxisOpts(name="Power", min_=0, max_=20),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="15%", pos_right="10%", orient="horizontal"),  # 图例放到右上方，横向排列
            datazoom_opts=opts.DataZoomOpts(is_show=True, type_="slider", range_start=0, range_end=96),  # 滑动条显示96个数据点
        )
        return line

def generate_visualization(model1_predictions, model2_predictions, actual_values, model1_errors, model2_errors, warning_threshold, theme_name, legend_names, time_data, width="1200px", height="600px"):
    """
    生成图表的函数，可以被其他模块调用
    :param model1_predictions: 模型1预测值
    :param model2_predictions: 模型2预测值
    :param actual_values: 实际值
    :param model1_errors: 模型1误差率
    :param model2_errors: 模型2误差率
    :param warning_threshold: 警戒线
    :param theme_name: 图表主题名称
    :param legend_names: 图例名称
    :param time_data: 时间数据
    :param width: 图表宽度
    :param height: 图表高度
    :return: HTML文件路径
    """
    # 创建PredictionData对象
    prediction_data = PredictionData(time_data, model1_predictions, model2_predictions, actual_values, model1_errors, model2_errors)

    # 创建可视化对象
    visualizer = Visualizer(prediction_data, warning_threshold, theme_name, legend_names, width, height)

    # 创建两个图表
    error_rate_chart = visualizer.plot_error_rate()
    prediction_vs_actual_chart = visualizer.plot_prediction_vs_actual()

    # 创建一个页面，将两个图表放到同一HTML文件中
    page = Page()
    page.add(error_rate_chart, prediction_vs_actual_chart)

    # 生成HTML文件名（替换空格和特殊字符为下划线，避免文件名错误）
    file_name = "web/zbcurve/"+re.sub(r'\W+', '_', theme_name) + ".html"

    # 渲染到HTML文件
    page.render(file_name)

    print(f"HTML文件已生成：{file_name}")
    return file_name

if __name__ == "__main__":
    import pandas as pd


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
    theme_name = "DRF3001 pred"  # 传入的主题名称
    legend_names = ["Model 1 Error Rate", "Model 2 Error Rate", 
                    "Model 1 Predictions", "Model 2 Predictions", "Actual Power"]

    # 调用生成图表的函数，并指定图表宽高
    generate_visualization(model1_predictions, model2_predictions, actual_values, 
                           model1_errors, model2_errors, warning_threshold, theme_name, 
                           legend_names, time_data, width="1600px", height="450px")