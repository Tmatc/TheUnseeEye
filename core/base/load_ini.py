from configparser import ConfigParser

class Configer():
    host = ""  # 主机ip
    user = ""  # 用户名
    password = ""  # 密码
    database = ""  # 连接的数据库
    port = 3306  # 数据库端口
    charset = "utf-8"  # 编码格式

    @staticmethod
    def load_cfg(ini_file:str):
        conf = ConfigParser()
        conf.read(ini_file)
        Configer.host = conf.get("database", "host")
        Configer.user = conf.get("database", "user")
        Configer.password = conf.get("database", "password")
        Configer.database = conf.get("database", "db")
        Configer.port = int(conf.get("database", "port"))
        Configer.charset = conf.get("database", "charset")
        