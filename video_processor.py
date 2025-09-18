import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch

import train_resnet
from logger import LogManager


class DiceVideoProcessor:
    def __init__(self,background=None,logger=None):
        self.background = background
        if logger is None:
            logger = LogManager.setup()
        self.logger=logger

        self.cnn = train_resnet.get_cnn_instance()
        self.background_frames = []  # 滑动窗口背景缓冲区
        self.background_angle_diff = 0
        self.last_frame = None
        self.last_dot = None
        self.last_second = None
        self.output_folder = 'images'
        self.features = []

    def _save_first_frame(self, video_path,output_folder='images'):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        angle = self.get_angle(frame)
        angle_diff = round(angle - 90)
        output_path = f'{output_folder}/frame0_{angle_diff}.jpg'
        if ret:
            cv2.imwrite(output_path, frame)
            print(f"首帧图片已保存到 {output_path}")
        else:
            raise ValueError("无法读取指定帧")

    def _extract_background(self, video_path, output_folder='images', num_frames=50, roi=None,second=0):
        if self.background is None:
            self._save_first_frame(video_path, output_folder)
        if len(self.background_frames) < num_frames:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            start = int(fps*second)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames = min(num_frames, total_frames)
            step = int(15*fps) #间隔15秒采样
            for i in range(start, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                if roi is not None:
                    x, y, w, h = roi
                    frame = frame[y:y + h, x:x + w]
                self.background_frames.append(frame)
                if len(self.background_frames)>=num_frames:
                    break
        # 初始化mean和M2为全零数组
        frames = self.background_frames
        mean = np.mean(frames, axis=0).astype(np.float32)
        std_dev = np.std(frames, axis=0).astype(np.float32)
        median_frame = np.median(frames, axis=0).astype(np.uint8)
        # 背景融合策略
        background = np.where(std_dev < 100, median_frame, mean).astype(np.uint8)
        background = cv2.medianBlur(background, 5)
        self.background = background

        sum_angles = 0
        for frame in frames:
            sum_angles += self.get_angle(frame)
        avg_angle = sum_angles / len(self.background_frames)
        self.background_angle_diff = round(avg_angle - 90)
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        background_path = f"output/background_{self.background_angle_diff}_{current_time}.jpg"
        cv2.imwrite(background_path, background)
        self.logger.info(f"background saved to {background_path}")
        self.background_frames.clear()
        cap.release()
        return self.background


    def detect_dice_feature(self, frame):
        dot, confidence = self.cnn.predict_image(frame)
        dice_roi,region = self.__extract_dice(frame)
        angle = self.get_angle(frame)
        angle_diff = round(angle-90)
        if angle_diff !=self.background_angle_diff:
            print(f"角度{angle_diff}和背景图片的{self.background_angle_diff}不一样")
        if dice_roi is not None:
            x1, y1, w1, h1 = region
            features = self._extract_features(dice_roi, x1 , y1 , w1 , h1 , dot)
            features = features.numpy() if isinstance(features, torch.Tensor) else features
            combined_features = np.concatenate((features.flatten(), [angle_diff]))
            return combined_features
        return None

    def _extract_features(self, dice_roi, x, y, w, h, dot):
        """从骰子区域提取特征"""
        # 转为灰度图
        gray = cv2.cvtColor(dice_roi, cv2.COLOR_BGR2GRAY)

        # 检测骰子点数（简化版，实际需要更复杂的算法）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.dilate(binary, kernel, iterations=1)

        # 计算中心点和角度
        moments = cv2.moments(binary)
        if moments["m00"] != 0:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x, center_y = w / 2, h / 2

        # 提取纹理特征 (LBP)
        # lbp_hist = self._extract_texture(dice_roi)

        # 提取形状特征 (Hu矩)
        hu_moments = self._extract_hu_moments(dice_roi)

        # 提取边缘特征 (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256]).flatten()

        # 计算角度矢量
        mu = hu_moments.tolist()
        if np.sum(mu) > 1e-5:  # 防止除零错误
            theta = 0.5 * np.arctan2(2 * mu[1], mu[0] - mu[2])
            angle_vector = [np.cos(theta), np.sin(theta)]
        else:
            angle_vector = [0.0, 0.0]
        # 返回特征向量
        features = {
            "position": (x, y),
            "size": (w, h),
            "center": (center_x, center_y),
            "dot": dot,
            "hu_moments": hu_moments.tolist(),
            "edge_hist": edge_hist.tolist(),
            "angle_vector": angle_vector
        }
        flat_features = [
            x, y, w, h, center_x, center_y, dot,
            *features["hu_moments"],
            *features["edge_hist"],
            *features["angle_vector"]
        ]

        return np.array(flat_features, dtype=np.float32)

    def get_angle(self,image):
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
            return 90

    def _extract_hu_moments(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            hu_moments = cv2.HuMoments(cv2.moments(max_contour)).flatten()
            return hu_moments
        return np.zeros(7)

    def process_video(self, video_path, roi=None,output_folder='train/images',step_second=17):
        """处理整个视频，提取骰子状态序列"""
        if self.background is None:
            self._extract_background(video_path, roi=roi)
        video_filename = os.path.basename(video_path)
        base, _ = os.path.splitext(video_filename)
        # output_folder = os.path.join(output_folder, video_filename.split('.')[0])
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_frame = None
        last_i = None
        last_dot = None
        cnt = 0
        for i in range(0, total_frames, fps * step_second):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]
            dot = self._recognize_dice_value(frame)
            if dot is None:
                cnt +=1
                if cnt>=3:
                    last_frame=None
                    last_dot = None
                continue
            else:
                cnt = 0
            if dot == 0:
                continue
            if last_dot is None:
                last_dot = dot
                last_frame = frame
                continue
            if dot != last_dot:
                if last_i is None or i - last_i > 25:
                    cv2.imwrite(f'{output_folder}/{dot}_{last_dot}-{i / fps}_{base}.jpg', last_frame)
                    last_i = i
            elif dot == last_dot:
                diff = cv2.absdiff(frame, last_frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                gray_diff[:80, :]=0
                _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                non_zero_pixels = cv2.countNonZero(thresh)
                if non_zero_pixels > 300:  # 假设100个像素的变化可以忽略
                    if last_i is None or i - last_i > 25:
                        dice_frame, poi = self.__extract_dice(frame)
                        if dice_frame is not None:
                            x1, y1, w1, h1 = poi
                            if 60 >= w1 >= 30 and 60 >= h1 >= 30:
                                cv2.imwrite(f'{output_folder}/{dot}_{last_dot}-{i / fps}_{base}.jpg', last_frame)
                                last_i = i
            last_frame = frame
            last_dot = dot
        cap.release()

    def _process_video(self, video_path, roi=None,output_folder='train/images',step_second=15):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        self._extract_background(video_path, roi=roi,second=0)
        next_update_background = 3600  # 1小时的帧数

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        second = 0
        n = step_second
        while True:
            second += n
            if second * fps > total_frames:
                # self.logger.info(f"已到达视频末尾，停止处理")
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * second))
            ret, frame = cap.retrieve()
            if not ret:
                # self.logger.info("Failed to read frame")
                break
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]
            if second>next_update_background:
                self._extract_background(video_path, roi=roi,second=second)
                next_update_background = next_update_background+3600
            second = self.next_frame(frame, second)
        cap.release()
        #将提取的features保存到文件
        video_filename = os.path.basename(video_path)
        base, _ = os.path.splitext(video_filename)

        # 转换为 NumPy 数组
        features_array = np.array(self.features, dtype=np.float32)
        folder_path = 'features'
        num = 0
        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_path} 不存在")
            num = 0
        else:
            npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            num = len(npy_files)
        self.save_features(features_array, f'features/{self.background_angle_diff}_{base}_{num+1}.npy')
        self.features  = []

    def next_frame(self, frame, second):
        """处理每一秒采样的帧图像"""
        dot, cf = self.cnn.predict_image(frame)
        if dot == 0:
            # self.logger.info(f"{second} 检测骰子在动: {dot}")
            return second + random.randint(2, 4)
        if cf < 0.97:
            self.logger.info(f"{second}检测骰子点数:{dot} 置信度 {cf:.4f} 太小 ")
            # cv2.imwrite(f"{self.image_dir}/{dot}_{second}_{cf:.4f}.jpg", frame)
            return second
        # self.logger.info(f"{second}检测骰子点数:{dot} 置信度 {cf:.4f}")
        if self.last_dot is None:
            self.last_dot = dot
            self.last_frame = frame
            return second
        changed = False
        if dot != self.last_dot:
            changed = True
            self.logger.info(f"{second}检测骰子点数变动: {self.last_dot} ==> {dot}")
            self.last_second = second
        if not changed and self.last_frame is not None:
            diff = cv2.absdiff(frame, self.last_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            gray_diff[0:80, :] = 0
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            non_zero_pixels = cv2.countNonZero(thresh)
            if non_zero_pixels > 200 and (self.last_second is None or second - self.last_second > 25):
                changed = True
                self.last_second = second
                self.logger.info(f"{second}检测骰子位置变动: {self.last_dot} ==> {dot}")
        if changed:
            self.extract_feature_and_save(self.last_frame,dot)
        self.last_frame = frame
        self.last_dot = dot
        return second

    def extract_simple_feature(self,frame):
        if frame is not None:
            dot, confidence = self.cnn.predict_image(frame)
            if confidence>=0.97:
                dice_roi, region = self.__extract_dice(frame)
                if region is not None:
                    angle = self.get_angle(frame)
                    x, y, w, h = region
                    angle_diff = round(angle - 90)
                    feature = [
                        x, y, w, h, dot, angle_diff
                    ]
                    return feature
        return None

    def extract_feature_and_save(self,last_frame,current_dot):
        if last_frame is not None:
            last_dot, confidence = self.cnn.predict_image(last_frame)
            dice_roi, region = self.__extract_dice(last_frame)
            if dice_roi is not None:
                angle = self.get_angle(last_frame)
                x, y, w, h = region
                angle_diff = round(angle - 90)
                feature = [
                    x, y, w, h, last_dot, angle_diff,current_dot
                ]
                flat_features = np.array(feature, dtype=np.float32)
                current_time = datetime.now().strftime("%Y%m%d%H%M%S")
                cv2.imwrite(
                    f"{self.output_folder}/{last_dot}_{current_dot}_{x}_{y}_{w}_{h}_{current_time}_{self.background_angle_diff}.jpg",
                    dice_roi)
                self.features.append(flat_features)
        return None

    def load_features(self, input_path):
        """
        从 .npy 文件中加载 features
        :param input_path: 输入文件路径
        :return: 加载的 NumPy 数组
        """
        # 确保文件存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        # 从 .npy 文件中加载数据
        features = np.load(input_path)
        print(f"Features loaded from {input_path}")
        return features

    def save_features(self, features, output_path):
        """
        将 features 保存到 .npy 文件
        :param features: 要保存的特征数据，通常是 NumPy 数组
        :param output_path: 输出文件路径
        """
        if not isinstance(features, np.ndarray):
            raise ValueError("features 必须是 NumPy 数组")

        # 确保输出目录存在
        if os.path.dirname(output_path)!='':
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存到 .npy 文件
        np.save(output_path, features)
        print(f"Features saved to {output_path}")

    def __extract_dice0(self,frame):
        """检测骰子的位置"""
        if self.background is None:
            raise ValueError("请先提取背景")
        # 计算当前帧与背景的差异
        diff = cv2.absdiff(frame, self.background)
        diff[0:80, :]=0
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 自适应直方图均衡化（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_diff)

        # Gamma亮度校正
        # gamma = 0.3  # 小于1时提升暗部亮度
        # brightened = np.power(enhanced / 255.0, gamma) * 255.0
        # brightened = brightened.astype(np.uint8)
        # cv2.imwrite(f"output/brightened{time.time()}.jpg", brightened)

        # 自适应阈值（结合OTSU算法）
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # cv2.imwrite(f"output/thresh{time.time()}.jpg", thresh)

        # 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 寻找轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 筛选符合条件的轮廓
            valid_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 60 >= w >= 30 and 60 >= h >= 30:
                    valid_contours.append(contour)

            if valid_contours:
                # 找到最大的轮廓（假设是骰子）

                max_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)

                # 提取骰子区域
                dice_roi = frame[y:y + h, x:x + w]
                # cv2.imwrite(f"output/dice_roi{time.time()}.jpg", dice_roi)
                return dice_roi,(x,y,w,h)
                # return brightened,(x,y,w,h)
        return None,None

    def __extract_dice(self, frame):
        """检测骰子的位置"""
        if self.background is None:
            raise ValueError("请先提取背景")

        # 计算当前帧与背景的差异
        diff = cv2.absdiff(frame, self.background)
        diff[0:80, :] = 0  # 裁剪顶部区域
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 自适应直方图均衡化（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_diff)

        # 自适应阈值（结合OTSU算法）
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 进一步去除投影（闭运算）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 调整核大小
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 筛选符合条件的轮廓
            valid_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 60 >= w >= 30 and 60 >= h >= 30:  # 宽高范围
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:  # 防止除零错误
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if 0.5 <= circularity <= 1.0:  # 筛选近似矩形的轮廓
                            valid_contours.append(contour)

            if valid_contours:
                # 找到最大的轮廓（假设是骰子）
                max_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)

                # 提取骰子区域
                dice_roi = frame[y:y + h, x:x + w]
                return dice_roi, (x, y, w, h)

        return None, None
    def _recognize_dice_value(self, frame,cnf=0.97):
        """识别骰子点数"""
        # 这里需要实现骰子点数识别算法
        # 简化版：根据检测到的点数确定骰子值

        cls,confidence =self.cnn.predict_image(frame)
        if confidence >= cnf:
            return int(cls)
        return None
