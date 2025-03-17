import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from train_resnet import CNN


class DiceVideoProcessor:
    def __init__(self):
        self.background = None
        self.dice_positions = []
        self.dice_results = []
        self.cnn = CNN()

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
    def _extract_background(self, video_path, output_folder='images', num_frames=100, roi=None):
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


    def detect_dice_state(self, frame):
        dice_roi,region = self._extract_craps(frame)
        x, y, w, h = region
        if dice_roi is not None:
            # # 提取特征（位置、大小、角度等）
            features = self._extract_features(dice_roi, x, y, w, h)
            return features
        return None

    def _extract_features(self, dice_roi, x, y, w, h):
        """从骰子区域提取特征"""
        # 转为灰度图
        gray = cv2.cvtColor(dice_roi, cv2.COLOR_BGR2GRAY)

        # 检测骰子点数（简化版，实际需要更复杂的算法）
        predicted_class, confidence = self.cnn.predict_image(dice_roi)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.dilate(binary, kernel, iterations=1)
        cv2.imwrite(f'output/binary_{time.time()}.jpg',binary)
        cv2.imwrite(f'output/gray_{time.time()}.jpg', gray)

        # 计算中心点和角度
        moments = cv2.moments(binary)
        if moments["m00"] != 0:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x, center_y = w / 2, h / 2

        # 提取纹理特征 (LBP)
        lbp_hist = self._extract_texture(dice_roi)

        # 提取形状特征 (Hu矩)
        hu_moments = self._extract_shape_features(dice_roi)

        # 返回特征向量
        return {
            "position": (x, y),
            "size": (w, h),
            "center": (center_x, center_y),
            "dots_count": predicted_class+1,
            "mean_color": np.mean(dice_roi, axis=(0, 1)).tolist(),
            "std_color": np.std(dice_roi, axis=(0, 1)).tolist(),
            "lbp_hist": lbp_hist.tolist(),
            "hu_moments": hu_moments.tolist()
        }

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

    def process_video(self, video_path, roi=None,output_folder='train/new_images'):
        """处理整个视频，提取骰子状态序列"""
        # if self.background is None:
        #     self._extract_background(video_path, roi=roi)
        video_filename = os.path.basename(video_path)
        base = video_filename.split('.')[0]
        # output_folder = os.path.join(output_folder, video_filename.split('.')[0])
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        save_frame_count = 0  # 记录总帧数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_frame = []
        last_dot = None
        for i in range(0, total_frames, fps*10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]
            dot = self._recognize_dice_value(frame)
            while dot==0:
                i = i+100
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                if roi is not None:
                    x, y, w, h = roi
                    frame = frame[y:y + h, x:x + w]
                dot = self._recognize_dice_value(frame)
            if dot is None:
                continue
            if last_dot is None:
                last_dot = dot
                last_frame.append(frame)
                continue
            if dot != last_dot:
                for j in range(len( last_frame)):
                    frame1 = last_frame[j]
                    output_path = f'{output_folder}/{dot}_{i/fps}_{base}-{j}.jpg'
                    cv2.imwrite(output_path, frame1)
                last_frame.clear()
            elif dot == last_dot:
                diff = cv2.absdiff(frame,last_frame)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                non_zero_pixels = cv2.countNonZero(thresh)
                if non_zero_pixels > 300:  # 假设100个像素的变化可以忽略
                    for j in range(len( last_frame)):
                        frame1 = last_frame[j]
                        output_path = f'{output_folder}/{dot}_{i/fps}_{base}-{j}.jpg'
                        cv2.imwrite(output_path, frame1)
                    last_frame.clear()
            last_frame.append(frame)
            last_dot = dot
        cap.release()

    def _process_video(self, video_path, roi=None):
        """处理整个视频，提取骰子状态序列"""
        if self.background is None:
            self.extract_background(video_path,roi=roi)

        cap = cv2.VideoCapture(video_path)
        is_moving = False
        last_features = None
        stable_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            features = self.detect_dice_state(frame, roi)

            if features is None:
                continue

            # 检测骰子是否在移动
            if last_features is not None:
                movement = self._calculate_movement(features, last_features)

                if movement > 5:  # 阈值可调整
                    is_moving = True
                    stable_frames = 0
                elif is_moving:
                    stable_frames += 1

                    # 骰子停止移动
                    if stable_frames > 10:  # 连续10帧稳定
                        is_moving = False
                        self.dice_positions.append(features)
                        # 识别骰子点数
                        dice_value = self._recognize_dice_value(frame)
                        self.dice_results.append(dice_value)

            last_features = features

        cap.release()
        return self.dice_positions, self.dice_results

    def _calculate_movement(self, current, previous):
        """计算两帧之间骰子的移动量"""
        curr_pos = current["position"]
        prev_pos = previous["position"]
        return np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)

    def _extract_craps(self,frame):
        """检测骰子的位置"""
        if self.background is None:
            raise ValueError("请先提取背景")
        # 计算当前帧与背景的差异
        diff = cv2.absdiff(frame, self.background)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 自适应直方图均衡化（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_diff)

        # Gamma亮度校正
        gamma = 0.3  # 小于1时提升暗部亮度
        brightened = np.power(enhanced / 255.0, gamma) * 255.0
        brightened = brightened.astype(np.uint8)
        # cv2.imwrite(f"output/brightened{time.time()}.jpg", brightened)

        # 自适应阈值（结合OTSU算法）
        _, thresh = cv2.threshold(brightened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # 根据实际情况调整面积范围
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.8 < aspect_ratio < 1.2:  # 骰子应接近正方形
                        valid_contours.append(contour)

            if valid_contours:
                # 找到最大的轮廓（假设是骰子）
                max_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)

                # 提取骰子区域
                dice_roi = frame[y:y + h, x:x + w]
                # cv2.imwrite(f"output/dice_roi{time.time()}.jpg", dice_roi)
                return dice_roi,(x,y,w,h)

        return None,None

    def _recognize_dice_value(self, frame,cnf=0.70):
        """识别骰子点数"""
        # 这里需要实现骰子点数识别算法
        # 简化版：根据检测到的点数确定骰子值

        cls,confidence =self.cnn.predict_image(frame)
        if confidence >= cnf:
            return int(cls)
        return None
