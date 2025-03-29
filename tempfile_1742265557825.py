import logging
import os
import shutil
import random


def copy_random_jpgs(src_folder, dest_folder, n):
    """
    从源文件夹中随机选择 n 个 .jpg 文件并复制到目标文件夹中。

    参数:
    src_folder (str): 源文件夹路径。
    dest_folder (str): 目标文件夹路径。
    n (int): 要复制的文件数量。
    """
    # 获取源文件夹中所有 .jpg 文件的列表
    jpg_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg')]

    # 检查源文件夹中是否有足够的 .jpg 文件
    if len(jpg_files) < n:
        raise ValueError(f"源文件夹中只有 {len(jpg_files)} 个 .jpg 文件，无法选择 {n} 个文件。")

    # 随机选择 n 个文件
    selected_files = random.sample(jpg_files, n)

    # 确保目标文件夹存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)

    # 复制选定的文件到目标文件夹
    for file_name in selected_files:
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        shutil.copy2(src_file, dest_file)
        logging.info(f"复制文件: {src_file} -> {dest_file}")


# 示例调用
if __name__ == "__main__":
    src_folder = 'train/new-images1'
    dest_folder = 'train/new-images_val'
    n = 2000  # 要复制的文件数量

    copy_random_jpgs(src_folder, dest_folder, n)
