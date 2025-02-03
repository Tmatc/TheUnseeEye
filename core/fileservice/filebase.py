import os
import shutil
from datetime import datetime


def backup_file_with_date(source_file, backup_dir, date_format="%Y%m%d%H%M"):
    """
    将单个源文件复制到备份目录，并在文件名后添加备份日期的后缀。
    
    :param source_file: 源文件的路径
    :param backup_dir: 备份目录的路径
    :param date_format: 日期格式，默认为 "YYYYMMDDHHmm"
    """
    try:
        # 检查源文件是否存在且为文件
        if not os.path.isfile(source_file):
            print(f"源文件不存在或不是一个文件: {source_file}")
            return
        
        # 创建备份目录（如果不存在）
        os.makedirs(backup_dir, exist_ok=True)
        print(f"备份目录已创建或已存在: {backup_dir}")
        
        # 获取当前日期字符串
        current_date = datetime.now().strftime(date_format)
        
        # 获取源文件的文件名和扩展名
        filename, ext = os.path.splitext(os.path.basename(source_file))
        
        # 构造备份文件的文件名
        backup_filename = f"{filename}_{current_date}{ext}"
        backup_file_path = os.path.join(backup_dir, backup_filename)
        
        # 复制文件
        shutil.copy2(source_file, backup_file_path)
        print(f"已复制文件: {source_file} 到 {backup_file_path}")
        # 确认复制成功后删除原始文件
        if os.path.exists(backup_file_path):
            os.remove(source_file)
            print(f"已删除原始文件: {source_file}")
        else:
            print("复制失败，未删除原始文件。")
    
    except Exception as e:
        print(f"备份过程中出现错误: {e}")



if __name__ == "__main__":
    source_file = "/path/to/source_file.txt"      # 源文件路径
    backup_dir = "/path/to/backup_directory"      # 备份目录路径
    backup_file_with_date(source_file, backup_dir)

