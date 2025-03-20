import os
import random
import cv2
import torch
import pandas as pd

# 定义文件夹路径
folder_path = 'train\\new_images-0'

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

# 加载图片
img = cv2.imread(image_path)
# img[:80,:]=0
img = img[80:,:]
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊以减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测提取边缘
edges = cv2.Canny(blurred, 50, 150)
# 使用YOLOv5进行推理
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
# results = model(img)
results = model(blurred)

# 获取检测结果
detections = results.pandas().xyxy[0]

for _, detection in detections.iterrows():
    x_min, y_min, x_max, y_max = map(int, detection[['xmin', 'ymin', 'xmax', 'ymax']])
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # 计算中心点
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

    print(f"骰子中心点：({center_x}, {center_y})")

# 显示结果
cv2.imshow('Frame', img)
# cv2.imshow('Blurred', blurred)
# cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
