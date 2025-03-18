import logging
import os
import random
import shutil


def move_random_images(source_dir, target_dir, move_ratio=0.2):
    """
    将源目录中的图片随机移动到目标目录。

    参数:
        source_dir (str): 源目录路径。
        target_dir (str): 目标目录路径。
        move_ratio (float): 移动的图片比例（0到1之间）。
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取源目录中的所有图片文件
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

    # 计算需要移动的图片数量
    num_images_to_move = int(len(image_files) * move_ratio)

    # 随机选择图片
    images_to_move = random.sample(image_files, num_images_to_move)

    # 移动图片
    for image in images_to_move:
        source_path = os.path.join(source_dir, image)
        target_path = os.path.join(target_dir, image)
        shutil.move(source_path, target_path)
        logging.info(f'Moved {image} from {source_dir} to {target_dir}')


# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义源目录和目标目录
source_directory = 'train/new_images'
target_directory = 'train/val'

# 移动图片（这里设置移动20%的图片）
move_random_images(source_directory, target_directory, move_ratio=0.2)
