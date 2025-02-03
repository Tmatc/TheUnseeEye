import pandas as pd
import time, datetime
import uuid


class FarmInfo():
    def __init__(self, file_info:str):
        self.data_info = pd.read_csv(file_info)
    
    # 插入js_farm_manager表的信息
    def base_info_f(self):
        turbine_nums = self.data_info['机组数量']
        ins_capas = self.data_info['装机容量']
        run_capas = self.data_info['运行容量']
        farm_names = self.data_info['场站名称']
        regions = self.data_info['所属区域']
        service_types = self.data_info['服务类型']
        service_status = self.data_info['服务状态']
        collection_ways = self.data_info['收资方式'] 
        zk_phones = self.data_info['主控电话']
        wind_types = self.data_info['风机型号']
        jdxs = self.data_info['集电线情况']
        use_nwps = self.data_info['使用的气象源']
        possess_nwps = self.data_info['拥有的气象源']

        basedata = []  # 场站级别
        basedata_jdx = []  # 集电线级别
        for i in range(len(farm_names)):
            tmp_fid = uuid.uuid4().hex
            tmp_turb = turbine_nums[i]
            tmp_ins = ins_capas[i]
            tmp_run = run_capas[i]
            tmp_farm = farm_names[i]
            tmp_region = regions[i]
            tmp_servicet = service_types[i]
            tmp_services = service_status[i]
            tmp_collection = collection_ways[i]
            tmp_zk = int(zk_phones[i])
            tmp_jdx = jdxs[i].split(",")
            tmp_windt = wind_types[i].split(",")
            tmp_usenwp = "%02d" % use_nwps[i]
            tmp_possess = possess_nwps[i]
            dt = time.localtime()
            tmp_ctime = datetime.datetime(dt.tm_year, dt.tm_mon, dt.tm_mday, dt.tm_hour, dt.tm_min, dt.tm_sec)
            basedata.append((tmp_fid, tmp_turb, tmp_ins, tmp_run, tmp_farm, tmp_region, tmp_servicet, tmp_services, tmp_collection, 
             tmp_zk, tmp_jdx, tmp_ctime))  # 插入js_farm_manager中的数据结构
            t_jdx = []  # 集电线级别
            for j in range(len(tmp_jdx)):
                jd = tmp_jdx[j]
                tt = tmp_windt[j]  # 单个集电线的设备型号和数量如xxx:20
                jdinfo = jd.split(":")
                t_jdx.append((uuid.uuid4().hex, tmp_fid, jdinfo[0], jdinfo[1], jdinfo[2], jdinfo[3], tt, tmp_usenwp, tmp_possess,
                 tmp_ctime))
            basedata_jdx.append(t_jdx)
        return basedata, basedata_jdx
    
    def equipment_info(self):
        names = self.data_info['名称']
        gecjs = self.data_info['生产厂家']
        models = self.data_info['型号']
        failure_rates = self.data_info['故障率']
 
        sbdata_fj = []  # 风机设备数据
        sbdata_gf = {}  # 光伏组件信息
        sbdata_nb = {}  # 逆变器信息

        t_data_gf1 = []
        t_data_gf2 = []
        t_data_nb1 = []
        t_data_nb2 = []
        for i in range(len(names)):
            tmp_sid = uuid.uuid4().hex
            tmp_name = names[i]
            tmp_gecj = gecjs[i]
            tmp_model = models[i]
            tmp_failure = failure_rates[i]
            if names[i] == "fj":
                tmp_fjb = f"espeed:{self.data_info['额定风速'][i]},epower:{self.data_info['额定功率'][i]},ylzj:{self.data_info['叶轮直径'][i]},lgheight:{self.data_info['轮毂高度'][i]},cutinspeed:{self.data_info['切入风速'][i]},cutoutspeed:{self.data_info['切出风速'][i]}"
                efficient = self.data_info['风能转换效率'][i]  # 效率，对应风电是风能转换效率，对应光伏是组件或逆变器转换效率
                sbdata_fj.append((tmp_sid, tmp_name, tmp_gecj, tmp_model, tmp_fjb, tmp_failure, efficient))
            elif names[i] == "gf":
                zid = uuid.uuid4().hex
                length = self.data_info['组件长度'][i]
                width = self.data_info['组件宽度'][i]
                efficient = self.data_info['转换效率'][i]
                maxpower = self.data_info['最大功率'][i]
                tempture = self.data_info['额定温度'][i]
                max_u = self.data_info['最大工作电压'][i]
                max_i = self.data_info['最大工作电流'][i]
                open_u = self.data_info['开路电压'][i]
                short_i = self.data_info['短路电流'][i]
                t_data_gf1.append((tmp_sid, tmp_name, tmp_gecj, tmp_model, tmp_failure))
                t_data_gf2.append((zid, tmp_sid, length, width, efficient, maxpower, tempture, max_u, max_i, open_u, short_i))
                sbdata_gf.update({"eqinfo": t_data_gf1, "zjinfo": t_data_gf2})
            elif names[i] == "nb":
                nid = uuid.uuid4().hex
                efficient = self.data_info['转换效率'][i]
                t_data_nb1.append((tmp_sid, tmp_name, tmp_gecj, tmp_model, tmp_failure))
                t_data_nb2.append((nid, tmp_sid, efficient))
                sbdata_nb.update({"eqinfo": t_data_nb1, "nbinfo": t_data_nb2})
        return sbdata_fj, sbdata_gf, sbdata_nb
    
    def base_power_curve(self, sid):
        wind_speed = self.data_info['wind_speed']
        power = self.data_info['expected_power']

        pdata = []
        for i in range(len(wind_speed)):
            id = uuid.uuid4().hex
            tmp_wspeed = wind_speed[i]
            tmp_power = power[i]
            pdata.append((id, sid, tmp_wspeed, tmp_power))
        return pdata
    

