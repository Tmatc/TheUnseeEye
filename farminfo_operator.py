import os
import uuid
from core.base import load_yml
from core.dbservice import farmdbo
from farminfo_manager.basefarmer import FarmInfo
from farminfo_manager.basenwp import DrOrganizationNwp


# 风电的基础信息入库
def base_farm_manager_fo1():
    baseinfofile = "information_container/farmbase/风电基础信息.csv"  # 电站基础信息文件
    farm_info = FarmInfo(baseinfofile)
    basefarminfo = farm_info.base_info_f()  # 获取进行信息数据，元组列表
    return {"farminfo_insert":basefarminfo[0], "farmjdx_insert_f": basefarminfo[1]}

def base_farm_manager_go1():
    pass

# 插入设备管理信息表
def equipment_info_f():  
    sbfile = "information_container/farmbase/设备信息表.csv"
    sb_info = FarmInfo(sbfile)
    sbdata = sb_info.equipment_info()
    return "sb_info_f", sbdata[0]

# 插入机组的基础功率曲线
def wind_turbine(sbdata):  
    powerfile = "information_container/farmdata"
    list_f = os.listdir(powerfile)
    sbinfos = sbdata  # 先获取设备基本信息

    pinfos = []  
    for i in range(len(list_f)):
        if list_f[i][:3] == "DRF":
            continue
        pcurvefile = os.path.join(powerfile, list_f[i]).replace("\\", "/")
        pcv = FarmInfo(pcurvefile)
        model = list_f[i].split("_")[0]  # 风机型号
        sid = ""
        for d in sbinfos:
            if model in d:
                sid = d[0]
        if len(sid) != 0:
            pdata = pcv.base_power_curve(sid)
            pinfos.append(pdata)
        else:
            raise Exception("没有符合的设备id,请检查文件信息")
    return "power_curve", pinfos

# 气象基本信息入库
def nwp_info(nwpinfo:dict,/):
    drorganwp = DrOrganizationNwp()
    nwpinfosql = drorganwp.nwpinfo_insert(nwpinfo)
    return nwpinfosql

# 将气象机构的数据插入数据库
def nwpdata_to_storage(fcode, nwpcode):
    '''
    fcode: 集电线的编号
    nwpcode: 气象编码
    '''
    nwpfile = "information_container/farmdata"
    for fname in os.listdir(nwpfile):
        if "nwp.csv" in fname and fname[:11] == fcode:
            nwpfpath = os.path.join(nwpfile, fname).replace("\\", "/")
            drorganwp = DrOrganizationNwp(nwpfile=nwpfpath)
            jid = farmdbo.select_id("jid", "js_jdx_manager", "farmcode", fcode)
            nwpid = farmdbo.select_id("nwp_id", "js_nwp_manager", "nwp_code", nwpcode)
            nwpdata = drorganwp.nwpdata_insert(jid, nwpid)
            return nwpdata
        

if __name__ == "__main__":

    # eq_pcv：风机设备信息以及机组出场功率曲线一起入库;
    # farm_basef: 风电场的基本信息入库
    # nwp_info: 执行气象机构的气象基本信息的入库
    # nwp_data: 执行将气象机构数据插入到数据库操作
    optype = "nwp_data"  # 执行的操作类型

    fcodejdx = "DRF3476J001"
    nwpcode = "06"

    mappfile = "mapping/db_farm_info_mapping.yml"  
    yml_data = load_yml.read_yaml_file(mappfile)  # 获取相应的sql语句
    import uuid
    nwpinfo = {"nwp_id": uuid.uuid4().hex, 
               "qx_name": "中科天机",
               "nwp_code": "13",
               "nwp_path": "/data/datasource/newZKTJ/nwp",
               "nwp_lenth": 1440,
               "nwp_ipp": "47.95.239.5"}

    match optype:
        case "farm_basef":
            keyinfo = base_farm_manager_fo1()
            basedata = keyinfo["farminfo_insert"]
            sql_farm_manager = yml_data["farminfo_insert"]  # 获取插入基础信息的sql
            sql_jdx_manager = yml_data["farmjdx_insert_f"]
            farmdbo.insert_base(sql_farm_manager, basedata)  # 插入场站管理表数据，对应js_farm_manager表
            for idx in keyinfo["farmjdx_insert_f"]:
                farmdbo.insert_base(sql_jdx_manager, idx)
        case "eq_pcv":
            keyeq, sbdata = equipment_info_f()
            eqsql = yml_data[keyeq]
            farmdbo.insert_base(eqsql, sbdata)  # 插入设备基础信息，对应js_equipment_manger表

            keypcv, pinfos = wind_turbine(sbdata)
            pcvsql = yml_data[keypcv]
            for d in pinfos:
                farmdbo.insert_base(pcvsql, d)  # 插入功率曲线
        case "nwp_info":
            nwpinfosql = nwp_info(nwpinfo)
            farmdbo.execete_sql(nwpinfosql)
        case "nwp_data":
            nwpdatasql = yml_data["nwpdata_insert"]
            nwpdata = nwpdata_to_storage(fcodejdx, nwpcode)
            farmdbo.insert_base(nwpdatasql, nwpdata)


    farmdbo.cursor.close()
    farmdbo.conn.close()
