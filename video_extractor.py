import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


class VideoExtractor:
    def __init__(self):
        self.background = None
        self.time_str=None
    def _process_frame(self, frame, mean, M2, frame_count):
        frame_float = frame.astype(np.float32)
        delta = frame_float - mean
        mean += delta / (frame_count + 1)
        delta2 = frame_float - mean
        M2 += delta * delta2
        return mean, M2

    def _save_first_frame(self, video_path,output_folder='images'):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        output_path = f'{output_folder}/frame0.jpg'
        if ret:
            cv2.imwrite(output_path, frame)
            print(f"首帧图片已保存到 {output_path}")
        else:
            raise ValueError("无法读取指定帧")
    def _calculate_median(self, cap, roi, n=100):
        """使用滑动窗口计算中位数（内存优化版）"""
        frames = []
        median = None
        step = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // n)

        for i in range(0, n * step, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                if roi:
                    x, y, w, h = roi
                    frame = frame[y:y + h, x:x + w]
                if len(frames) < 30:  # 滑动窗口保持30帧
                    frames.append(frame)
                else:
                    frames[i % 30] = frame  # 循环覆盖旧帧
                median = np.median(frames, axis=0).astype(np.uint8) if frames else None

        return median
    def _extract_background(self, video_path,output_folder='images',num_frames=100, roi=None):
        self._save_first_frame(video_path, output_folder)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = min(num_frames, total_frames)

        # 使用累积法代替全帧存储
        mean = None
        M2 = None
        frame_count = 0

        # 随机采样帧（减少重复区域影响）
        step = max(1, total_frames // num_frames)
        frames = []
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]
            frames.append(frame)
        # 初始化mean和M2为全零数组
        if frames:
            first_frame = frames[0].astype(np.float32)
            mean = np.zeros_like(first_frame)
            M2 = np.zeros_like(first_frame)

        with ThreadPoolExecutor() as executor:
            futures = []
            for frame in frames:
                future = executor.submit(self._process_frame, frame, mean.copy(), M2.copy(), frame_count)
                futures.append(future)
            for idx, future in enumerate(futures):
                mean, M2 = future.result()
                frame_count += 1
        # 计算标准差
        std_dev = np.sqrt(M2 / (frame_count - 1)) if frame_count > 1 else np.zeros_like(mean)

        # 使用中位数代替均值（更抗噪）
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        median_frame = self._calculate_median(cap, roi, n=num_frames)

        # 背景融合策略
        background = np.where(std_dev < 50, median_frame, mean).astype(np.uint8)
        background = cv2.medianBlur(background, 5)
        self.background = background
        cv2.imwrite(f"{output_folder}/background{time.time()}.jpg", background)
        cap.release()
        return self.background

    def extract_images(self, video_path, frame_step=100, output_folder='images',roi=None):
        # self.background = self._extract_background(video_path, output_folder, num_frames=100, roi=roi)
        self.time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            relative_time = i / fps
            self.save_image(frame,relative_time=relative_time,roi=roi,output_folder=output_folder)

    def save_image(self, frame, relative_time,roi=None,output_folder='images'):
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y:y + h, x:x + w]

        cv2.imwrite(f"{output_folder}/s_{relative_time}_{self.time_str}.jpg", frame)
        # """检测骰子的位置和状态"""
        # if self.background is None:
        #     raise ValueError("请先提取背景")
        # # 计算当前帧与背景的差异
        # diff = cv2.absdiff(frame, self.background)
        # gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        #
        # # 阈值处理
        # # 自适应直方图均衡化（CLAHE）
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # enhanced = clahe.apply(gray_diff)
        #
        # # Gamma亮度校正
        # gamma = 0.4  # 小于1时提升暗部亮度
        # brightened = np.power(enhanced / 255.0, gamma) * 255.0
        # brightened = brightened.astype(np.uint8)
        # # cv2.imwrite(f"{output_folder}/g_{relative_time}_{self.time_str}.jpg", brightened)
        #
        # # 自适应阈值（结合OTSU算法）
        # _, thresh = cv2.threshold(brightened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #
        # # 形态学开运算去噪
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # # cv2.imwrite(f"output/thresh{time.time()}.jpg", thresh)
        #
        # # # 寻找轮廓
        # # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # # 寻找轮廓
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #
        # if contours:
        #     # 筛选符合条件的轮廓
        #     valid_contours = []
        #     for contour in contours:
        #         area = cv2.contourArea(contour)
        #         if 100 < area < 10000:  # 根据实际情况调整面积范围
        #             x, y, w, h = cv2.boundingRect(contour)
        #             aspect_ratio = float(w) / h
        #             if 0.8 < aspect_ratio < 1.2:  # 骰子应接近正方形
        #                 valid_contours.append(contour)
        #
        #     if valid_contours:
        #         # 找到最大的轮廓（假设是骰子）
        #         max_contour = max(valid_contours, key=cv2.contourArea)
        #         x, y, w, h = cv2.boundingRect(max_contour)
        #         # # 提取骰子区域
        #         dice_roi = frame[y:y + h, x:x + w]
        #         cv2.imwrite(f"{output_folder}/r_{relative_time}_{self.time_str}.jpg", dice_roi)
        #         return dice_roi
        #         # # 创建一个与原图像大小相同的黑色图像
        #         # mask = np.zeros_like(frame)
        #         # # 在黑色图像上绘制矩形区域，填充为白色
        #         # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)
        #         # # 使用掩膜提取 max_contour 区域的颜色
        #         # dice_roi = cv2.bitwise_and(frame, mask)
        #         # # 保存处理后的图像
        #         # cv2.imwrite(f"{output_folder}/m_{relative_time}_{self.time_str}.jpg", dice_roi)
        #
        # return None

if __name__ == "__main__":
    extractor = VideoExtractor()
    video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\video_1741004387.flv"
    # video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2.flv"
    # video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2-202503131755.flv"
    video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2-202503142204.flv"
    video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2-202503150745.flv"
    video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2202503151933.flv"
    video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2-03161025.flv"
    video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2-03152236.flv"
    video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2-03161446.flv"
    video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\B21_2-03162110.flv"
    roi = [514, 134, 224, 224]
    output_folder = 'images'
    frame_step = 600
    os.makedirs(output_folder, exist_ok=True)
    extractor.extract_images(video_path=video_path, frame_step=frame_step, output_folder=output_folder,roi=roi)

