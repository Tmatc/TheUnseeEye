from core.resultservice.sysdata import SysData
from core.base.timeseries import TimeSeriesProcessor



if __name__ == "__main__":
    import time
    fcodejdx = "DRF3007J001"  # 集电线
    usenwp = "06"
    starttime = "2017-07-03 00:00:00"  # 数据开始时间
    endtime = "2025-07-30 23:45:00"  # 数据结束时间
    needfield = "DateTime, humidity70, pressure70, temperature70, direction70, speed70"
    start = time.time()
    sys = SysData(fcodejdx=fcodejdx)(usenwp, needfield, starttime, endtime).data_align()
    # sys = SysData(fcodejdx=fcodejdx)
    print("==================实际功率=================")
    print(sys["realjpower"])
    print("==================气象数据=================")
    print(sys["nwpdata"])
    print("==================功率曲线=================")
    print(sys["powercurve"])
    print("==================训练数据集=================")
    print(sys.dataset_renwp)
    processor = TimeSeriesProcessor(time_column='dtime', value_columns=['real_power'], time_interval='15T', 
                                    missing_threshold=2000, interpolation_method='linear')
    serdata = processor.process_data(sys.dataset_renwp)
    print("=====================时续数据=================")
    print(serdata['final_data'])
    end = time.time()
    print("总耗时:", end-start)