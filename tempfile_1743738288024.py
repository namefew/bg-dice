import os

import cv2
import numpy as np

def get_angle(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义优化的紫色范围
    lower_purple = np.array([120, 60, 60])  # 扩大H通道范围
    upper_purple = np.array([160, 255, 255])

    # 提取紫色区域
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # 形态学优化（使用椭圆核进行闭运算）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 查找紫色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 确保找到至少一个轮廓
    if len(contours) > 0:
        # 取最大的轮廓（通常是骰钟罩子的轮廓）
        largest_contour = max(contours, key=cv2.contourArea)

        # 拟合椭圆
        ellipse = cv2.fitEllipse(largest_contour)

        # 解析椭圆参数
        center, axes, angle = ellipse
        center_x, center_y = center
        major_axis, minor_axis = axes
        rotation_angle = angle

        # print(f"椭圆中心: ({center_x:.2f}, {center_y:.2f})")
        # print(f"半长轴: {major_axis / 2:.2f}")
        # print(f"半短轴: {minor_axis / 2:.2f}")
        # print(f"旋转角度: {rotation_angle:.2f} 度")

        # 绘制拟合的椭圆
        image_with_ellipse = image.copy()
        cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)

        # 显示结果
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Purple Mask', mask)
        # cv2.imshow('Detected Ellipse', image_with_ellipse)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return rotation_angle

    else:
        print("未检测到紫色区域的轮廓！")
        return None

files = os.listdir('output')
for file in files:
    if file.endswith('.jpg'):
       angle = get_angle('output/' + file)
       print(f"{file} 倾斜角度： {round(angle-90)}  {angle-90:.4f}")
