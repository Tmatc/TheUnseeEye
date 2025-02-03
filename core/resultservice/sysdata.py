# 此模块主要讲数据库中的全场功率数据封装到类，方便调用
import os
import pandas as pd
from core.base import load_yml
from core.dbservice import farmdbo, datadbo
from datetime import datetime, timedelta


tf = os.getcwd()
conf = os.path.join(tf, "mapping/db_select_mapping.yml").replace("\\", "/")  # sql语句映射配置文件路径


class SysData():
    '''
    该类主要作用是封装数据库中的 "集电线级别" 和 "场站级别" 的 "系统功率数据"， "天气预报数据"
    根据指定的集电线或场站编号和指定的时间范围来获取
    完成选取数据在时间上时对应的；
    同时具备选择和判断时间连续的数据的功能
    '''
    def __init__(self, *,fcode = None, fcodejdx=None):
        ymldata = load_yml.read_yaml_file(conf)
        self.fcode = fcode  # 场站编号
        self.fcodejdx = fcodejdx  # 集电线编号
        t = fcodejdx[:3] if fcodejdx is not None else fcode[:3]
        self.realpjdxsql = ymldata['js_syspower_jdx_f'] if t == "DRF" else ymldata['js_syspower_jdx_g'] # 查询指定集电线的实际功率数据对象sql
        self.realpsql = ymldata['js_syspower_f'] if t == "DRF" else ymldata['js_syspower_g'] # 查询指定场站的实际功率数据对象sql
        self.nwpsql = ymldata['js_weather']  # 查询某个集电线的某个气象机构的气象数据sql

        self.realjdxpower = None
        self.realpower = None
        self.nwpdata = None  # 气象数据
        self.powerdata  = None  # 功率曲线

    def __call__(self, nwpcode="06", selectfield="DateTime, humidity70, pressure70, temperature70, direction70, speed70", stime = "", etime = ""):
        try:
            jid = farmdbo.select_id("jid", "js_jdx_manager", "farmcode", self.fcodejdx)  # 集电线id
            f_id = farmdbo.select_id("f_id", "js_farm_manager", "jdx", self.fcodejdx)  # 场站id
            nwp_id = farmdbo.select_id("nwp_id", "js_nwp_manager", "nwp_code", nwpcode)  # 气象源id
            nwpsql = self.nwpsql % (selectfield, jid, nwp_id, stime, etime)
            if self.fcode is None and self.fcodejdx is not None:
                realpjdxsql = self.realpjdxsql % (jid, stime, etime)
                self.realjdxpower = pd.DataFrame(datadbo.execete_sql(realpjdxsql))  # 场站某条集电线对应的实际功率数据
            
            elif self.fcode is not None:
                realpsql = self.realpsql % (f_id, stime, etime)
                self.realpower = pd.DataFrame(datadbo.execete_sql(realpsql))  # 场站对应的实际功率数据情况
            
            elif self.fcode is None and self.fcodejdx is None:
                raise Exception("请配置您的集电线或场站信息!!")

            self.nwpdata = pd.DataFrame(datadbo.execete_sql(nwpsql))  # 气象数据

            v = farmdbo.select_id("sbtype", "js_jdx_manager", "farmcode", self.fcodejdx)
            sbtype = v.split(":")[0]  # 设备型号
            sid = farmdbo.select_id("sid", "js_equipment_manger", "model", sbtype)
            pv = datadbo.get_power_curve(sid)
            self.powerdata = pd.DataFrame(pv)

            return self
        except Exception as e:
            print(f"{__name__}",e)

    def __getitem__(self, key):
        if key == "powercurve":
            return self.powerdata
        elif key == "realjpower":
            return self.realjdxpower
        elif key == "realpower":
            return self.realpower
        elif key == "nwpdata":
            return self.nwpdata


    def __del__(self):
        try:
            print("正在关闭数据库连接")
            farmdbo.cursor.close()
            farmdbo.conn.close()
            datadbo.cursor.close()
            datadbo.conn.close()
        except Exception as e:
            print(e)
    
    def data_align(self):
        '''得到一个含功率标签的数据集'''
        if self.realjdxpower is not None and self.nwpdata is not None:
            tmp = self.realjdxpower[["dtime", 'real_power']]
            df = pd.merge(tmp, self.nwpdata, left_on="dtime", right_on="DateTime")
            del df['DateTime']
            self.dataset_renwp = df  # 训练数据集
        else:
            raise Exception("提供的数据不能组成一个含标签的数据集，请检查数据库数据")
        return self
    
    def timeseries(self):
        '''得到一个时间连续的数据集'''
        pass

    def generate_time_series(self,start_time_str, end_time_str, interval_minutes=15, time_format="%Y-%m-%d %H:%M:%S"):
        """
        生成指定时间间隔的连续时间序列。

        :param start_time_str: 开始时间字符串
        :param end_time_str: 结束时间字符串
        :param interval_minutes: 时间间隔（分钟）
        :param time_format: 时间字符串的格式
        :return: 时间序列列表
        """
        start_time = datetime.strptime(start_time_str, time_format)
        end_time = datetime.strptime(end_time_str, time_format)
        interval = timedelta(minutes=interval_minutes)

        time_series = []
        current_time = start_time
        while current_time <= end_time:
            time_series.append(current_time)
            current_time += interval

        return time_series




        
                       