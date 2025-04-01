import cv2
import numpy as np


def crop_top_80_if_needed(img):
    height, width = img.shape[:2]
    if height == 224 and width == 224:
        return img[100:, :, :]
    return img


def detect_dice(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return []

    cut = 0
    height, width = img.shape[:2]
    if height == 224 and width == 224:
        cut = 100
        img = img[cut:, :, :]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 骰子的颜色范围（红色）
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dice_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = cv2.contourArea(cnt)

        # 筛选骰子轮廓
        if 0.9 <= aspect_ratio <= 1.1 and area > 100:
            dice_rects.append((x, y, w, h))

    return dice_rects


def adaptive_crop(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return None

    dice_rects = detect_dice(image_path)

    if not dice_rects:
        print("No dices detected.")
        return None

    # 计算所有骰子的边界框的最小外接矩形
    min_x = min(rect[0] for rect in dice_rects)
    min_y = min(rect[1] for rect in dice_rects)
    max_x = max(rect[0] + rect[2] for rect in dice_rects)
    max_y = max(rect[1] + rect[3] for rect in dice_rects)

    # 添加一些边距以确保骰子完全包含在内
    margin = 10
    min_x = max(0, min_x - margin)
    min_y = max(0, min_y - margin)
    max_x = min(img.shape[1], max_x + margin)
    max_y = min(img.shape[0], max_y + margin)

    cropped_img = img[min_y:max_y, min_x:max_x]

    return cropped_img


if __name__ == "__main__":
    image_paths = [
        'train/images-1/2_3780.0_20250314203236.jpg',
        'train/images-1/3_765.0_20250314193525.jpg',
        'train/images-1/4_2565.0_20250314212630.jpg',
        'train/images-1/5_2290.0_20250314203236.jpg'
    ]

    for image_path in image_paths:
        cropped_img = adaptive_crop(image_path)
        if cropped_img is not None:
            cv2.imshow('Cropped Image', cropped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()