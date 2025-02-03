# 此模块主要完成从数据库系统拿取数据
import pymysql
import os
import pymysql.cursors
from core.base.load_ini import Configer


tf = os.getcwd()
conf = os.path.join(tf, "setting/sys_cfg.ini").replace("\\", "/")  # 数据库信息配置文件路径
Configer.load_cfg(conf)


conn = pymysql.connect(
    host= Configer.host,
    user= Configer.user,
    password= Configer.password,
    database= Configer.database,
    port= Configer.port,
    charset= Configer.charset,
    autocommit=True,
    cursorclass=pymysql.cursors.DictCursor
)
cursor = conn.cursor()

# 执行给定的sql
def execete_sql(sql):
    try:
        cursor.execute(sql)
        print(sql, "::执行成功！")
        return cursor.fetchall()
    except Exception as e:
        print(e)

def get_power_curve(sid):
    try:
        sql = f"SELECT wind_speed, power FROM js_wind_turbine WHERE sid='%s' ORDER BY wind_speed" % sid
        cursor.execute(sql)
        return cursor.fetchall()
    except Exception as e:
        print(f"{__name__}:40行::", e)

