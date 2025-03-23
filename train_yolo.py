
import os
from pathlib import Path

# 定义数据目录
images_dir = 'yolov5/train/yolo/images'
labels_dir = 'yolov5/train/yolo/labels'

# 确保标签目录存在
os.makedirs(labels_dir, exist_ok=True)

# 遍历所有图像文件
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg'):
        # 解析文件名获取标注信息
        parts = filename.split('_')
        dot_number = 0
        x_center = float(parts[2]) / 224  # 假设图像宽度为640
        y_center = float(parts[3]) / 224  # 假设图像高度为480
        width = float(parts[4]) / 224
        height = float(parts[5].split('.')[0]) / 224

        # 写入.txt文件
        label_filename = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))
        with open(label_filename, 'w') as f:
            f.write(f"{dot_number} {x_center} {y_center} {width} {height}\n")

print("标注文件生成完成！")

# 训练模型
train_path = Path('yolov5/train.py').resolve()
data_yaml_path = Path('data.yaml').resolve()
weights_path = Path('yolov5s.pt').resolve()

# 使用命令行方式进行训练
import subprocess
subprocess.run(['python', 'yolov5/train.py', '--img', '224', '--batch', '16', '--epochs', '100', '--data', str(data_yaml_path), '--weights', str(weights_path)])