import pandas as pd
import os
from core.fileservice import filebase


'''该模块主要负责对各种原始文件进行整理成指定格式的标准文件，包含对场站功率数据，不同的气象数据'''
class Dremt(object):
    '''该类是处理从原dremt数据库中导出的数据结构'''
    def __init__(self, flist):
        self.file_list = flist  # 文件路径列表
    
    # 系统功率数据组合
    def data_merge_gl(self, neighborhoodspeed, outpath):
        if len(self.file_list) % 2 != 0:
            raise Exception("一个场站必须只有偶数个文件，一个是存储实际功率的，一个是存储实测气象的")
        if len(self.file_list) == 2:
            df1 = pd.read_csv(self.file_list[0]) if ".csv" in self.file_list[0] else pd.read_excel(self.file_list[0])
            df2 = pd.read_csv(self.file_list[1]) if ".csv" in self.file_list[1] else pd.read_excel(self.file_list[1])
            if 'layer' in df2.columns:
                df2 = df2[df2['layer']==neighborhoodspeed]
            if 'layer' in df1.columns:
                df1 = df1[df1['layer']==neighborhoodspeed]
            syspdata = pd.merge(df1, df2, on='Date_Time')
            syspdata['Date_Time'] = syspdata["Date_Time"].str.replace("_", " ")
            new_data = pd.DataFrame({"日期和时间": syspdata['Date_Time'], "实际功率": syspdata['RealPower'], 
                                    "测风塔风速": syspdata['speed'], "测风塔温度": syspdata['temperature'], 
                                    "测风塔湿度": syspdata['humidity'], "测风塔风向": syspdata['direction']})
            file_out_path = os.path.join(outpath, self.file_list[0].split("/")[-1][:11]+"_syspower.csv").replace("\\", "/")
            print(len(new_data))
       
            new_data.to_csv(file_out_path, index=False)
            print(f"文件{file_out_path}标准化完成，一条集电线")
            filebase.backup_file_with_date(self.file_list[0], "information_container/back/dataother")
            filebase.backup_file_with_date(self.file_list[1], "information_container/back/dataother")
    
        if len(self.file_list) == 4:
            pass
    
    def datafp(self, outpath):
        for fpfile in self.file_list:
            fpdf = pd.read_csv(fpfile)
            fpdf["Date_Time"] = fpdf["Date_Time"].str.replace("_", " ")
            fpfilename = os.path.join(outpath, fpfile.split("/")[-1][:11]+"_fp.csv").replace("\\", "/")
            fpdf.to_csv(fpfilename, index=False)
            filebase.backup_file_with_date(fpfile, "information_container/back/dataother")
    

