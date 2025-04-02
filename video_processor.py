import os
import time
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

import train_resnet

class DiceVideoProcessor:
    def __init__(self,background=None):
        self.background = background
        self.dice_positions = []
        self.dice_results = []
        self.cnn = train_resnet.get_cnn_instance()

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


    def _extract_background(self, video_path, output_folder='images', num_frames=20, roi=None):
        self._save_first_frame(video_path, output_folder)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = min(num_frames, total_frames)
        # 使用累积法代替全帧存储
        mean = None
        M2 = None
        frame_count = 0
        # 随机采样帧（减少重复区域影响）
        step = int(total_frames/num_frames)
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
            if len(frames)>=num_frames:
                break
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
        # 获取视频文件的basename
        video_basename = os.path.basename(video_path)
        video_name, _ = os.path.splitext(video_basename)
        # 保存背景图像时包含视频文件的basename
        output_path = f"{output_folder}/background_{video_name}.jpg"
        cv2.imwrite(output_path, background)
        cap.release()
        return self.background


    def detect_dice_feature(self, frame):
        dot, confidence = self.cnn.predict_image(frame)
        dice_roi,region = self.__extract_dice(frame)
        if dice_roi is not None:
            x1, y1, w1, h1 = region
            w = frame.shape[1]
            h = frame.shape[0]
            # # 提取特征（位置、大小、角度等）
            # features0 = self.cnn.extract_features_from_image(frame)
            features = self._extract_features(dice_roi, x1 , y1 , w1 , h1 , dot)
            # features0 = features0.numpy() if isinstance(features0, torch.Tensor) else features0
            features = features.numpy() if isinstance(features, torch.Tensor) else features
            # 合并特征数组
            # combined_features = np.concatenate((features0.flatten(), features))
            #
            # return combined_features
            return features
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
        hu_moments = self._extract_shape_features(dice_roi)

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

    def _extract_texture(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lbp = self.local_binary_pattern(gray)
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        return hist

    def local_binary_pattern(self, image, P=8, R=1, method='uniform'):
        """
        计算图像的局部二值模式 (LBP) 特征。

        参数:
            image: 输入灰度图像。
            P: 邻域采样点数，默认为8。
            R: 邻域半径，默认为1。
            method: LBP 方法类型，默认为 'uniform'。

        返回:
            lbp_image: 计算得到的 LBP 图像。
        """
        # 确保输入图像为灰度图像
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 初始化 LBP 图像
        lbp_image = np.zeros_like(image)

        # 获取图像尺寸
        height, width = image.shape

        # 遍历图像中的每个像素
        for y in range(R, height - R):
            for x in range(R, width - R):
                center = image[y, x]
                pattern = []  # 显式初始化 pattern 为列表

                # 计算邻域内的像素值
                for i in range(P):
                    angle = 2 * np.pi * i / P
                    x_neighbor = int(x + R * np.cos(angle))
                    y_neighbor = int(y + R * np.sin(angle))

                    if x_neighbor >= 0 and x_neighbor < width and y_neighbor >= 0 and y_neighbor < height:
                        neighbor = image[y_neighbor, x_neighbor]
                        pattern.append(1 if neighbor >= center else 0)
                    else:
                        pattern.append(0)

                # 将二进制模式转换为整数
                lbp_value = sum([pattern[i] << i for i in range(P)])

                # 处理 'uniform' 方法
                if method == 'uniform':
                    # 使用更简洁的方式计算 transitions
                    pattern_shifted = pattern[1:] + pattern[:1]
                    transitions = sum(np.array(pattern) != np.array(pattern_shifted))
                    if transitions <= 2:
                        lbp_value = sum([pattern[i] << i for i in range(P)])
                    else:
                        lbp_value = P + 1

                lbp_image[y, x] = lbp_value

        return lbp_image

    def _extract_shape_features(self, roi):
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
                if last_i is None or i - last_i > 20:
                    cv2.imwrite(f'{output_folder}/{dot}_{last_dot}-{i / fps}_{base}.jpg', last_frame)
                    last_i = i
            elif dot == last_dot:
                diff = cv2.absdiff(frame, last_frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                gray_diff[:80, :]=0
                _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                non_zero_pixels = cv2.countNonZero(thresh)
                if non_zero_pixels > 300:  # 假设100个像素的变化可以忽略
                    if last_i is None or i - last_i > 10:
                        dice_frame, poi = self.__extract_dice(frame)
                        if dice_frame is not None:
                            x1, y1, w1, h1 = poi
                            if 60 >= w1 >= 30 and 60 >= h1 >= 30:
                                cv2.imwrite(f'{output_folder}/{dot}_{last_dot}-{i / fps}_{base}.jpg', last_frame)
                                last_i = i
            last_frame = frame
            last_dot = dot
        cap.release()

    def _process_video(self, video_path, roi=None,output_folder='train/images',step_second=17):
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
        frame0 = None
        last_frame = None
        last_i = None
        last_dot = None
        for i in range(0, total_frames, fps * step_second):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]
            dot = self._recognize_dice_value(frame)
            if dot == 0 or dot is None:
                continue
            if last_dot is None:
                last_dot = dot
                last_frame = frame
                continue
            if dot != last_dot:
                if last_i is None or i - last_i > 20:
                    dice_frame,poi = self.__extract_dice(last_frame)
                    if dice_frame is not None:
                        x1,y1,w1,h1=poi
                        if 60 >= w1 >= 30 and 60 >= h1 >= 30:
                            cv2.imwrite(f'{output_folder}/../images0/{dot}_{last_dot}-{x1}_{y1}_{w1}_{h1}_{i / fps}_{base}.jpg', dice_frame)
                            cv2.imwrite(f'{output_folder}/../images/{dot}_{last_dot}-{x1}_{y1}_{w1}_{h1}_{i / fps}_{base}.jpg',last_frame)

                            # features0 = self.cnn.extract_features_from_image(last_frame)
                            features = self._extract_features(dice_frame, x1, y1 , w1 , h1, last_dot)
                            # features0 = features0.numpy() if isinstance(features0, torch.Tensor) else features0
                            features = features.numpy() if isinstance(features, torch.Tensor) else features
                            classify = dot-1
                            # 合并特征数组
                            combined_features = np.concatenate(( features, [classify]))
                            # 保存合并后的特征向量
                            feature_file_path = f'{output_folder}/{dot}_{last_dot}-{x1}_{y1}_{w1}_{h1}_{i / fps}_{base}.npy'
                            np.save(feature_file_path, combined_features)
                    last_i = i
            elif dot == last_dot:
                diff = cv2.absdiff(frame, last_frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                gray_diff[:80, :]=0
                _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                non_zero_pixels = cv2.countNonZero(thresh)
                if non_zero_pixels > 300:  # 假设100个像素的变化可以忽略
                    if last_i is None or i - last_i > 10:
                        dice_frame0, poi0 = self.__extract_dice(frame)
                        if poi0 is not None:
                            x0, y0, w0, h0 = poi0
                            if 60 >= w0 >= 30 and 60 >= h0 >= 30:
                                dice_frame, poi = self.__extract_dice(last_frame)
                                if dice_frame is not None:
                                    x1, y1, w1, h1 = poi
                                    if 60 >= w1 >= 30 and 60 >= h1 >= 30:
                                        # cv2.imwrite(f'{output_folder}/{dot}_{x1}_{y1}_{w1}_{h1}_{i / fps}_{base}.jpg', frame)
                                        cv2.imwrite(
                                            f'{output_folder}/../images0/{dot}_{last_dot}-{x1}_{y1}_{w1}_{h1}_{i / fps}_{base}.jpg',
                                            dice_frame)
                                        cv2.imwrite(
                                            f'{output_folder}/../images/{dot}_{last_dot}-{x1}_{y1}_{w1}_{h1}_{i / fps}_{base}.jpg',
                                            last_frame)

                                        # features0 = self.cnn.extract_features_from_image(last_frame)
                                        features = self._extract_features(dice_frame, x1, y1, w1 , h1,
                                                                          last_dot)
                                        # features0 = features0.numpy() if isinstance(features0,torch.Tensor) else features0
                                        features = features.numpy() if isinstance(features, torch.Tensor) else features
                                        classify = dot - 1
                                        # 合并特征数组
                                        combined_features = np.concatenate(( features, [classify]))
                                        # 保存合并后的特征向量
                                        feature_file_path = f'{output_folder}/{dot}_{last_dot}-{x1}_{y1}_{w1}_{h1}_{i / fps}_{base}.npy'
                                        np.save(feature_file_path, combined_features)
                last_i = i
            last_frame = frame
            last_dot = dot
        cap.release()



    def _calculate_movement(self, current, previous):
        """计算两帧之间骰子的移动量"""
        curr_pos = current["position"]
        prev_pos = previous["position"]
        return np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)

    def _extract_dice(self, frame):
        """检测骰子的位置"""
        if self.background is None:
            raise ValueError("请先提取背景")
        # 计算当前帧与背景的差异
        diff = cv2.absdiff(frame, self.background)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 自适应直方图均衡化（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_diff)
        return enhanced,None

    def __extract_dice(self,frame):
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

    def _recognize_dice_value(self, frame,cnf=0.97):
        """识别骰子点数"""
        # 这里需要实现骰子点数识别算法
        # 简化版：根据检测到的点数确定骰子值

        cls,confidence =self.cnn.predict_image(frame)
        if confidence >= cnf:
            return int(cls)
        return None
