import os

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
        x_center = (float(parts[1]) +float(parts[3])/2)/ 224  # 假设图像宽度为640
        y_center = (float(parts[2])+float(parts[4])/2) / 224  # 假设图像高度为480
        width = float(parts[3]) / 224
        height = float(parts[4]) / 224

        # 写入.txt文件
        label_filename = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))
        with open(label_filename, 'w') as f:
            f.write(f"{dot_number} {x_center} {y_center} {width} {height}\n")

print("标注文件生成完成！")
