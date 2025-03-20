import os
import random
import cv2
import numpy as np

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

# 读取图像
image = cv2.imread(image_path)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊以减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测提取边缘
edges = cv2.Canny(blurred, 50, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 拟合椭圆
for contour in contours:
    area = cv2.contourArea(contour)
    if 1000 < area < 10000:  # 根据实际情况调整面积范围
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)

# 提取红色骰子区域
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 70, 50])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def calculate_relative_position(ellipse_center, dice_bbox):
    ellipse_x, ellipse_y = ellipse_center
    dice_x, dice_y, dice_w, dice_h = dice_bbox
    relative_x = (dice_x + dice_w / 2) - ellipse_x
    relative_y = (dice_y + dice_h / 2) - ellipse_y
    return relative_x, relative_y

for contour in contours:
    area = cv2.contourArea(contour)
    if 100 < area < 1000:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:
                ellipse = cv2.fitEllipse(contour)
                ellipse_center = ellipse[0]
                relative_x, relative_y = calculate_relative_position(ellipse_center, (x, y, w, h))
                print(f"Relative position: ({relative_x}, {relative_y})")

cv2.imshow('Detected Dice', image)
cv2.imshow('Edges', edges)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
