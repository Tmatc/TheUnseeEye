from core.fileservice.down_up import CloudFileTransfer


if __name__ == '__main__':
# 初始化类并连接云服务器
    cloud = CloudFileTransfer()
    
    # 上传文件
    # cloud.upload_file("D:/workspace/work/tmp/Meteologica_DRF1152_2025020517_weather.txt", 
    #                   "/data/")
    
    # 下载单个文件
    cloud.download_file("/data/datasource/newxby/nwp/2025/02/DRF1152/", 
                        "D:/workspace/work/tmp/")
    
    # 下载整个目录
    # cloud.download_file("/remote/path/remote_directory", "local_downloaded_directory")
    
    # 关闭连接
    cloud.close()