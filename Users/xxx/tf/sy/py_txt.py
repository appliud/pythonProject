import os


def change_file_extensions(directory, old_ext, new_ext):
    """
    更改指定目录下所有文件的后缀名。

    :param directory: 要处理的目录路径。
    :param old_ext: 要替换的旧后缀（例如 '.txt'）。
    :param new_ext: 新的后缀（例如 '.csv'）。
    """
    # 确保目录路径以斜杠结尾
    if not directory.endswith(os.sep):
        directory += os.sep

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 获取文件的完整路径
        file_path = os.path.join(directory, filename)

        # 检查是否为文件且具有指定的旧后缀
        if os.path.isfile(file_path) and filename.endswith(old_ext):
            # 构造新的文件名
            new_filename = filename[:-len(old_ext)] + new_ext
            new_file_path = os.path.join(directory, new_filename)

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} -> {new_file_path}')


if __name__ == "__main__":
    # 指定要处理的目录路径
    directory = "./txt"

    # 指定要替换的旧后缀和新的后缀
    old_extension = ".py"  # 例如 .txt
    new_extension = ".txt"  # 例如 .md

    # 调用函数进行文件后缀名更改
    change_file_extensions(directory, old_extension, new_extension)