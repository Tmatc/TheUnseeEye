import uuid
import pandas as pd


class DrOrganizationNwp(object):
    '''
    此类主处理气象机构订阅的气象数据
    '''
    def __init__(self, nwpfile=""):
        self.nwpdata = None
        if len(nwpfile) != 0:
            self.farmcode = nwpfile.split("/")[-1][:11]  # 集电线编号
            self.nwpdata = pd.read_csv(nwpfile)
    
    def nwpinfo_insert(self, nwpinfo):
        sql = "INSERT INTO js_nwp_manager (nwp_id, qx_name, nwp_code, nwp_path, nwp_lenth, nwp_ipp) VALUES "+\
        "('{nwp_id}', '{qx_name}', '{nwp_code}', '{nwp_path}', {nwp_lenth}, '{nwp_ipp}')".format_map(nwpinfo)
        return sql
    def nwpdata_insert(self, jid, nwpid):
        if self.nwpdata is None:
            raise Exception("气象数据是空的!!")
        datetime = self.nwpdata["Date_Time"]
        humidity10 = self.nwpdata["humidity10"]
        humidity30 = self.nwpdata["humidity30"]
        humidity50 = self.nwpdata["humidity50"]
        humidity60 = self.nwpdata["humidity60"]
        humidity70 = self.nwpdata["humidity70"]
        humidity80 = self.nwpdata["humidity80"]
        humidity90 = self.nwpdata["humidity90"]
        humidity100 = self.nwpdata["humidity100"]
        humidity130 = self.nwpdata["humidity130"]
        humidity140 = self.nwpdata["humidity140"]
        pressure10 = self.nwpdata["pressure10"]
        pressure30 = self.nwpdata["pressure30"]
        pressure50 = self.nwpdata["pressure50"]
        pressure60 = self.nwpdata["pressure60"]
        pressure70 = self.nwpdata["pressure70"]
        pressure80 = self.nwpdata["pressure80"]
        pressure90 = self.nwpdata["pressure90"]
        pressure100 = self.nwpdata["pressure100"]
        pressure130 = self.nwpdata["pressure130"]
        pressure140 = self.nwpdata["pressure140"]
        temperature10 = self.nwpdata["temperature10"]
        temperature30 = self.nwpdata["temperature30"]
        temperature50 = self.nwpdata["temperature50"]
        temperature60 = self.nwpdata["temperature60"]
        temperature70 = self.nwpdata["temperature70"]
        temperature80 = self.nwpdata["temperature80"]
        temperature90 = self.nwpdata["temperature90"]
        temperature100 = self.nwpdata["temperature100"]
        temperature130 = self.nwpdata["temperature130"]
        temperature140 = self.nwpdata["temperature140"]
        direction10 = self.nwpdata["direction10"]
        direction30 = self.nwpdata["direction30"]
        direction50 = self.nwpdata["direction50"]
        direction60 = self.nwpdata["direction60"]
        direction70 = self.nwpdata["direction70"]
        direction80 = self.nwpdata["direction80"]
        direction90 = self.nwpdata["direction90"]
        direction100 = self.nwpdata["direction100"]
        direction130 = self.nwpdata["direction130"]
        direction140 = self.nwpdata["direction140"]
        speed10 = self.nwpdata["speed10"]
        speed30 = self.nwpdata["speed30"]
        speed50 = self.nwpdata["speed50"]
        speed60 = self.nwpdata["speed60"]
        speed70 = self.nwpdata["speed70"]
        speed80 = self.nwpdata["speed80"]
        speed90 = self.nwpdata["speed90"]
        speed100 = self.nwpdata["speed100"]
        speed130 = self.nwpdata["speed130"]
        speed140 = self.nwpdata["speed140"]
        nwpdata = []
        for i in range(len(datetime)):
            nwpdata.append((uuid.uuid4().hex, jid, nwpid, self.farmcode, datetime[i],humidity10[i],humidity30[i],
             humidity50[i], humidity60[i], humidity70[i], humidity80[i], humidity90[i], humidity100[i],
             0, humidity130[i], humidity140[i], 0, 0, pressure10[i], pressure30[i],pressure50[i],
             pressure60[i], pressure70[i], pressure80[i], pressure90[i], pressure100[i], 0, pressure130[i],
             pressure140[i], 0, 0, temperature10[i], temperature30[i], temperature50[i], temperature60[i],
             temperature70[i], temperature80[i], temperature90[i], temperature100[i], 0, temperature130[i],
             temperature140[i], 0, 0, direction10[i], direction30[i], direction50[i], direction60[i],
             direction70[i],direction80[i],direction90[i],direction100[i],0,direction130[i], direction140[i],
             0, 0, speed10[i], speed30[i], speed50[i], speed60[i], speed70[i], speed80[i], speed90[i],
             speed100[i], 0, speed130[i], speed140[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        return nwpdata