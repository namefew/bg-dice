import os
import shutil
import re

def parse_filename(filename):
    """
    解析文件名，提取 base 和 i/fps
    """
    match = re.match(r'(\d+)_(\d+)_(\d+\.\d+)_(.+)\.jpg', filename)
    if match:
        dot = int(match.group(1))
        save_frame_count = int(match.group(2))
        timestamp = float(match.group(3))
        base = match.group(4)
        return base, timestamp
    return None, None

def move_files(source_folder, target_folder):
    """
    移动符合条件的文件
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    files = os.listdir(source_folder)
    file_dict = {}

    # 解析文件名并分组
    for file in files:
        base, timestamp = parse_filename(file)
        if base is not None and timestamp is not None:
            if base not in file_dict:
                file_dict[base] = []
            file_dict[base].append((file, timestamp))

    # 找出符合条件的文件并移动
    for base, file_list in file_dict.items():
        file_list.sort(key=lambda x: x[1])  # 按 timestamp 排序
        i = 0
        while i < len(file_list) - 1:
            file1, timestamp1 = file_list[i]
            file2, timestamp2 = file_list[i + 1]
            if abs(timestamp2 - timestamp1) == 5:
                source_path1 = os.path.join(source_folder, file1)
                source_path2 = os.path.join(source_folder, file2)
                target_path1 = os.path.join(target_folder, file1)
                target_path2 = os.path.join(target_folder, file2)
                try:
                    shutil.move(source_path1, target_path1)
                    shutil.move(source_path2, target_path2)
                    print(f"Moved {file1} to {target_path1}")
                    print(f"Moved {file2} to {target_path2}")
                    # 从文件列表中移除已处理的文件
                    file_list.pop(i + 1)
                    file_list.pop(i)
                except FileNotFoundError as e:
                    print(f"Error moving files: {e}")
            else:
                i += 1

if __name__ == "__main__":
    source_folder = 'train/new_images'
    source_folder = 'C:/Users/fred/Desktop/new_images'
    target_folder = 'C:/Users/fred/Desktop/fix_image'
    move_files(source_folder, target_folder)
