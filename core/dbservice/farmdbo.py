import pymysql
import os
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
    autocommit=True
)
cursor = conn.cursor()

def insert_base(sql:str, data:list):
    try:
        cursor.executemany(sql, data)
        print(f"{sql}::执行成功")
    except Exception as e:
        print(e)

# 根据一个指定的条件查询某个表的主键id
def select_id(id, tablename, concol, convalue):
    '''
    id: 要查询的id
    tablename: 表名称
    concol: 条件列
    convalue: 条件列的取值
    '''
    
    sql = f"SELECT {id} FROM {tablename} WHERE {concol} LIKE '{convalue}%';"
    try:
        cursor.execute(sql)
        v = cursor.fetchone()
        
        return v[0] if v is not None else 0
    except Exception as e:
        print(e)

# 执行给定的sql
def execete_sql(sql):
    try:
        cursor.execute(sql)
        print(sql, "::执行成功！")
    except Exception as e:
        print(e)