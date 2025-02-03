import os
import uuid
import pandas as pd
from core.dbservice import farmdbo
from core.fileservice import filebase


'''该模块只负责处理标准化后的文件'''
class SysPower():
    def __init__(self, normalflist):
        self.normaldataf = normalflist
    
    # 从标准系统功率文件获取数据进行入库
    def getdata_from_syspfile(self):
        sysdata = []
        if len(self.normaldataf) == 1:
            fid = farmdbo.select_id("f_id", "js_farm_manager", "jdx", os.path.basename(self.normaldataf[0])[:11])
            jid = farmdbo.select_id("jid", "js_jdx_manager", "farmcode",os.path.basename(self.normaldataf[0])[:11])
            df = pd.read_csv(self.normaldataf[0])
            for i in range(len(df)):
                sysdata.append((uuid.uuid4().hex, fid, jid, df.iloc[i,0], round(df.iloc[i,1],8), round(df.iloc[i,2],8), round(df.iloc[i,3], 8), round(df.iloc[i,4], 8), round(df.iloc[i,5],8)))
        return sysdata
    
    # 从标准预测功率文件获取数据进行入库，该文件来源于其他系统导出的预测功率
    def getprepower_fpfile_outer(self):
        relist = []  # 一个三维数组，最外层的索引表示哪条集电线
        for fpfile in self.normaldataf:
            fpdata = pd.read_csv(fpfile)
            fcode = fpfile.split("/")[-1][:11]
            fplist = []
            for i in range(len(fpdata)):
                fplist.append((uuid.uuid4().hex, fcode, fpdata.iloc[i,0], fpdata.iloc[i,1]))
            relist.append(fplist)
            filebase.backup_file_with_date(fpfile, "information_container/back/farmdata")
        return relist


            
