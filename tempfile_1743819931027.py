import os

import numpy as np


def load_features_by_prefix(folder_path='features'):
    # 初始化三个列表暂存不同类别的特征
    zero_list = []
    negative_list = []
    positive_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)

            # 提取前缀数字（基于你的原始实现）
            try:
                num = int(filename.split('_')[0])
                feature = np.load(file_path)

                if num == 0:
                    zero_list.append(feature)
                elif num < 0:
                    negative_list.append(feature)
                else:
                    positive_list.append(feature)
            except (ValueError, IndexError):
                continue  # 忽略不符合命名规范的文件

    # 将列表转换为numpy数组（按行堆叠）
    return (
        np.vstack(zero_list) if zero_list else np.array([]),
        np.vstack(negative_list) if negative_list else np.array([]),
        np.vstack(positive_list) if positive_list else np.array([])
    )

zero_arr, pos_arr,neg_arr = load_features_by_prefix()
print(zero_arr.shape, neg_arr.shape, pos_arr.shape)