import pandas as pd
import numpy as np
from algorithmapp.cnn_lstm_appv01 import CnnLstmWindPowerApp


# ====================== 测试入口 ======================
def main():
    # --------------------
    # 模拟原始数据（增加数据量以适应滑动窗口）
    # --------------------
    # 原始数据列: [dtime, real_power, humidity70, pressure70, temperature70, direction70, speed70]
    df_raw = pd.DataFrame({
        'dtime': pd.date_range(start='2025-01-01 00:00', periods=50, freq='H').strftime('%Y-%m-%d %H:%M'),
        'real_power': np.random.randint(50, 1200, size=50),
        'humidity70': np.random.randint(30, 60, size=50),
        'pressure70': np.random.randint(1000, 1020, size=50),
        'temperature70': np.random.randint(15, 30, size=50),
        'direction70': np.random.randint(50, 180, size=50),
        'speed70': np.random.randint(5, 20, size=50)
    })

    # --------------------
    # 模拟理论功率数据（根据风速和功率关系）
    # --------------------
    df_theory = pd.DataFrame({
        'wind_speed': [5, 10, 15, 20, 25],
        'power':      [55, 210, 440, 790, 1000]
    })

    # --------------------
    # 初始化应用 (window_size=3, feature_num=5)
    # --------------------
    app = CnnLstmWindPowerApp(
        window_size=3,
        feature_num=5,
        model_path='windpower_cnn_lstm_model.pt',
        batch_size=16,  # 调整为16以适应较小的数据量
        learning_rate=1e-3,
        epochs=20,  # 减少训练轮数以加快测试
        early_stop_patience=3,
        save_best_model=True
    )

    # --------------------
    # A) 训练模式
    # --------------------
    print("===== 训练模式 =====")
    try:
        # 选择数据清洗方法：'cluster' 或 'theory'
        cleaning_method = 'cluster'  # 或 'theory'
        app.train(df_raw, df_theory=df_theory, cleaning_method=cleaning_method)
    except Exception as e:
        print(f"[ERROR] 训练过程中发生错误: {e}")

    # --------------------
    # B) 预测模式
    # --------------------
    print("\n===== 预测模式 =====")
    # 假设预测时原始数据不包含 real_power
    df_predict = pd.DataFrame({
        'dtime': pd.date_range(start='2025-01-03 14:00', periods=10, freq='H').strftime('%Y-%m-%d %H:%M'),
        'humidity70': np.random.randint(30, 60, size=10),
        'pressure70': np.random.randint(1000, 1020, size=10),
        'temperature70': np.random.randint(15, 30, size=10),
        'direction70': np.random.randint(50, 180, size=10),
        'speed70': np.random.randint(5, 20, size=10)
        # real_power 缺失
    })

    # 检查预测数据是否有缺失值
    print("[DEBUG] 预测数据缺失值统计:")
    print(df_predict.isnull().sum())

    try:
        df_predicted = app.predict(df_predict)
        print("[预测结果]")
        print(df_predicted)

        # 若需要评估，将预测结果与真实值进行合并
        # 假设真实值在df_raw中对应的时间点
        # 例如，预测结果对应的dtime为 '2025-01-03 18:00', '2025-01-03 19:00', '2025-01-03 20:00'
        # 根据实际数据生成真实值
        real_times = df_predicted['dtime'].tolist()
        df_real_power = df_raw[df_raw['dtime'].isin(real_times)][['dtime', 'real_power']]
        df_evaluation = pd.merge(df_predicted, df_real_power, on='dtime', how='left')

        # 评估
        metrics = app.evaluate(df_evaluation)
        print("[评估结果]")
        print(metrics)
    except Exception as e:
        print(f"[ERROR] 预测过程中发生错误: {e}")


if __name__ == "__main__":
    main()