
import os

def count_files_by_prefix(folder_path):
    """ 统计文件夹中以 1-6 开头的文件数量 """
    counts = {str(i): 0 for i in range(0, 7)}

    # 遍历文件夹中的所有 .jpg 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            prefix = filename.split('_')[0]
            if prefix in counts:
                counts[prefix] += 1

    return counts

# 指定文件夹路径
folder_path = 'train/images'

# 统计文件数量
file_counts = count_files_by_prefix(folder_path)

# 打印结果
for prefix, count in file_counts.items():
    print(f'Files starting with {prefix}: {count}')