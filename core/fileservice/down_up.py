import paramiko
import os
from core.base.load_ini import Configer

class CloudFileTransfer:
    '''
    该类主要提供从云服务器下载文件和上传文件到云服务器的功能
    下载文件时支持下载单个文件或整个目录，上传文件时只支持上传单个文件
    
    '''
    def __init__(self):
        """
        初始化时连接到云服务器
        :param hostname: 云服务器地址
        :param port: 云服务器端口
        :param username: 云服务器用户名
        :param password: 云服务器密码
        """
        Configer.load_remote_file("setting/sys_cfg.ini")
        self.hostname = Configer.host
        self.port = int(Configer.port)
        self.username = Configer.user
        self.password = Configer.password
        self.client = None
        self.sftp = None
        self.connect()

    def connect(self):
        """建立与云服务器的SSH连接"""
        try:
            # 创建 SSH 客户端实例
            self.client = paramiko.SSHClient()
            # 自动添加主机到 known_hosts
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # 连接到云服务器
            self.client.connect(self.hostname, username=self.username, password=self.password, port=self.port)
            # 打开 SFTP 会话
            self.sftp = self.client.open_sftp()
            print(f"成功连接到 {self.hostname}")
        except Exception as e:
            print(f"连接失败: {e}")
            raise

    def upload_file(self, local_filepath, remote_dir):
        """
        上传文件到云服务器
        :param local_filepath: 本地文件路径
        :param remote_dir: 云服务器目标目录路径
        """
        try:
            # 检查本地文件是否存在
            if not os.path.isfile(local_filepath):
                print(f"本地文件 {local_filepath} 不存在!")
                return False
            # 获取文件名（从路径中提取）
            filename = os.path.basename(local_filepath)
            # 生成目标文件的完整路径
            remote_filepath = os.path.join(remote_dir, filename)

            # 上传文件
            self.sftp.put(local_filepath, remote_filepath)
            # 检查上传是否成功
            if self._check_file_exists(remote_filepath):
                print(f"文件 {local_filepath} 上传成功到 {remote_filepath}")
                return True
            else:
                print(f"文件上传失败: {remote_filepath} 不存在")
                return False
        except Exception as e:
            print(f"上传失败: {e}")
            return False

    def download_file(self, remote_filepath, local_dir):
        """
        从云服务器下载文件，支持下载单个文件或整个目录
        :param remote_filepath: 云服务器文件路径
        :param local_dir: 本地目录路径
        """
        try:
            # 获取文件名（从路径中提取）
            filename = os.path.basename(remote_filepath)
            # 生成本地保存文件的路径
            local_filepath = os.path.join(local_dir, filename)

            # 检查远程路径是文件还是目录
            if self._is_directory(remote_filepath):
                # 如果是目录，递归下载
                print(f"开始下载目录 {remote_filepath} 到 {local_filepath}")
                self._download_directory(remote_filepath, local_filepath)
            else:
                # 如果是文件，直接下载
                print(f"开始下载文件 {remote_filepath} 到 {local_filepath}")
                self.sftp.get(remote_filepath, local_filepath)
                # 检查下载是否成功
                if os.path.isfile(local_filepath):
                    print(f"文件 {remote_filepath} 下载成功到 {local_filepath}")
                else:
                    print(f"文件下载失败: {local_filepath} 不存在")
        except Exception as e:
            print(f"下载失败: {e}")

    def _is_directory(self, path):
        """检查路径是否为目录"""
        try:
            # 检查文件模式是否为目录模式 16877（目录的模式）
            return self.sftp.stat(path).st_mode == 16877
        except FileNotFoundError:
            return False

    def _download_directory(self, remote_dir, local_dir):
        """递归下载目录"""
        try:
            # 获取远程目录下所有文件和子目录
            file_list = self.sftp.listdir_attr(remote_dir)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            for file_attr in file_list:
                remote_file = remote_dir + '/' + file_attr.filename
                local_file = os.path.join(local_dir, file_attr.filename)
                if self._is_directory(remote_file):
                    # 如果是目录，递归下载
                    self._download_directory(remote_file, local_file)
                else:
                    # 如果是文件，下载文件
                    self.sftp.get(remote_file, local_file)
                    print(f"下载文件 {remote_file} 到 {local_file}")
        except Exception as e:
            print(f"下载目录失败: {e}")

    def _check_file_exists(self, filepath):
        """检查文件是否存在"""
        try:
            self.sftp.stat(filepath)
            return True
        except FileNotFoundError:
            return False

    def close(self):
        """关闭与云服务器的连接"""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
        print(f"连接到 {self.hostname} 已关闭")


if __name__ == "__main__":
    # 初始化类并连接云服务器
    cloud = CloudFileTransfer(hostname="47.95.239.5", 
                              port=22, 
                              username="root", password="DREy5185747")
    
    # 上传文件
    cloud.upload_file("D:/workspace/work/tmp/Meteologica_DRF1152_2025020517_weather.txt", 
                      "/data/")
    
    # 下载单个文件
    # cloud.download_file("/data/datasource/newxby/nwp/2025/02/DRF1152/", 
    #                     "D:/workspace/work/tmp")
    
    # 下载整个目录
    # cloud.download_file("/remote/path/remote_directory", "local_downloaded_directory")
    
    # 关闭连接
    cloud.close()
