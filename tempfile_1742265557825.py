import os
import random
import cv2
import numpy as np
import torch
import pandas as pd

# 定义文件夹路径
folder_path = 'train\\features'

# 获取文件夹中的所有文件名
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 随机选择一个文件
if files:
    random_file = random.choice(files)
    image_path = os.path.join(folder_path, random_file)
else:
    print("文件夹为空")
    exit()

print(f"随机选择的文件：{image_path}")


features = np.load(image_path)
print(f'长度：{len(features)}，data：{features}')