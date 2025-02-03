import os
from core.base.load_ini import Configer
from core.base import load_yml
from core.dbservice import farmdbo
from collection_manager.datanormalizer.format_syspower import Dremt
from collection_manager.datanormalizer.format_nwp import OrganizationNwp
from collection_manager.normal_tostorage import SysPower


# 收资文件类型的过滤分类
def file_filter(file_directory:str, fcode:str):
    file_dict = {}
    dremtfs = []  # dremt库csv文件筛选后的实际功率和测风塔文件列表
    v1sz = []  # v1系统的人工收资
    dremtnwp = []  # dremt库csv文件nwp数据列表
    dremtfp = []  # dremt库csv文件fp数据列表

    normal_sysp = []  # 标准系统功率文件列表
    normal_turbinep = []  # 标准机组数据列表
    for f in os.listdir(file_directory):
        bff = f.split("_")
        if bff[-1] in ("rp.csv", "wt.csv") and f[:7] == fcode:
            dremtfs.append(os.path.join(file_directory, f).replace("\\", "/"))
        if bff[-1] == "nwp.csv" and f[:7] == fcode:
            dremtnwp.append(os.path.join(file_directory, f).replace("\\", "/"))
        if bff[-1] == "fp.csv" and f[:7] == fcode:
            dremtfp.append(os.path.join(file_directory, f).replace("\\", "/"))
        if ("测风塔信息" in f or "功率报表" in f) and f[:7] == fcode: # 针对v1系统的收资判断
            v1sz.append(os.path.join(file_directory, f).replace("\\", "/"))
        if "syspower" in f and f[:7] == fcode:
            normal_sysp.append(os.path.join(file_directory, f).replace("\\", "/"))
        if "turbine" in f and f[:7] == fcode:
            normal_turbinep.append(os.path.join(file_directory, f).replace("\\", "/"))
    
        # 若后期还有其他类型可以继续添加
        file_dict.update({"dremt":dremtfs, "v1sz": v1sz, "syspower":normal_sysp, "turbine": normal_turbinep,
                          "dremtnwp": dremtnwp, "dremtfp": dremtfp})  
    return file_dict

# 根据原文件选择不同的文件标准化方法
def normalization_selection(fd, fc, neibor=80, outpath=""):
    '''
    fd: 文件所在目录
    fc: 场站编码，如drf3007
    neibor: 测风塔或气象等具备层高属性的数据的临近轮毂高度的层高
    outpath: 标准文件存放的目录
    '''
    file_dict = file_filter(fd, fc)
    for k in file_dict:
        if k == "dremt":
            t_file = file_dict.get(k)
            if len(t_file) == 0:
                continue
            dremt = Dremt(t_file)
            dremt.data_merge_gl(neibor, outpath)
        if k == "v1sz":
            pass
        if k == "dremtnwp":  # 执行将drmt库导出的csv气象文件标准化
            t_file = file_dict.get(k)
            if len(t_file) == 0:
                continue
            organwp = OrganizationNwp(t_file)
            organwp.dremt_nwp_normal(outpath)
        if k == "dremtfp":  # 执行将drmt库导出的csv预测功率文件标准化
            t_file = file_dict.get(k)
            if len(t_file) == 0:
                continue
            dremt = Dremt(t_file)
            dremt.datafp(outpath)



# 对标准功率文件执行入库操作
def normalfile_to_storage(fc):
    normal_file_path = "information_container/farmdata"
    file_dict = file_filter(normal_file_path,fc)
    ndict = {}
    syslist = []  # 系统功率信息列表
    turbinelist = []  # 机组信息
    fplist = []  # 预测功率文件列表
    for k in file_dict:
        if k == "syspower":
            t_file = file_dict.get(k)
            if len(t_file) == 0:
                continue
            syspower = SysPower(t_file)
            sysdata = syspower.getdata_from_syspfile()
            syslist.extend(("farmpower_f", sysdata))
        if k == "turbine":
            pass
        if k == "dremtfp":
            t_file = file_dict.get(k)
            if len(t_file) == 0:
                continue
            syspower = SysPower(t_file)
            relist = syspower.getprepower_fpfile_outer()
            fplist.extend(("other_predict_pv", relist))

        ndict.update({"powercwt": syslist, "turbine": turbinelist, "dremtfp": fplist})

    return ndict


if __name__ == "__main__":
    # normal: 执行文件标准化操作
    # normal_to_syspower: 执行将系统功率标准文件入库操作
    operation = "normal_to_syspower"  

    fcode = "DRF3476"  # 指定操作的场站名称
    neighborhoodspeed = 70  # 测风塔或气象等具备层高属性的数据的临近轮毂高度的层高
    
    # f = "setting/sys_cfg.ini"
    original_file = "information_container/dataother"  # 未标准化的文件目录
    outpath = "information_container/farmdata"  # 标准化后的文件目录
    mappfile = "mapping/db_put_in_storage_mapping.yml"
    yml_data = load_yml.read_yaml_file(mappfile)


    match operation:
        case "normal":
            normalization_selection(original_file, fcode, neighborhoodspeed, outpath)
        case "normal_to_syspower":
            info_type = normalfile_to_storage(fcode)
            for k in info_type:
                sysinfo = info_type[k]
                if len(sysinfo) == 0:
                    continue
                syssql = yml_data[sysinfo[0]]
                if isinstance(sysinfo[1][0], list):  # sysinfo[1]是数据列表，如果第一层的元素是list说明他不是一个站或集电线数据
                    for d in sysinfo[1]:
                        farmdbo.insert_base(syssql, d)
                else:
                    farmdbo.insert_base(syssql, sysinfo[1])


    farmdbo.cursor.close()
    farmdbo.conn.close()