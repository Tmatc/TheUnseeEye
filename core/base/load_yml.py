import yaml
import os


def read_yaml_file(file_path:str):
    """
    :param file_path: str, YAML 文件路径
    :return: dict 或 list, 返回解析后的 YAML 数据
    :raises: FileNotFoundError, ValueError
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 确保文件扩展名为 .yaml 或 .yml
    if not file_path.endswith(('.yaml', '.yml')):
        raise ValueError("Invalid file extension. Please provide a .yaml or .yml file.")

    try:
        # 读取文件并安全加载
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    try:
        # YAML 文件路径
        yaml_file_path = "c.yml"
        yaml_data = read_yaml_file(yaml_file_path)
        print("YAML Data:", yaml_data)
        print()
        print(yaml_data['farmpower_f'])
    except Exception as e:
        print(f"Error: {e}")
