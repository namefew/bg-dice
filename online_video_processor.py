import os
import traceback
import random
from datetime import datetime

import cv2
import numpy as np

import config
import train_resnet
import threading


class DiceOnlineVideoProcessor:
    def __init__(self, roi, logger=None):
        self.url = None
        self.is_seekable = True
        self.background = None
        self.background_angle_diff = 0
        self.running = False
        self.cap = None
        self.fps = None
        self.roi = roi
        self.dot_cnn = train_resnet.get_cnn_instance()
        self.last_dot = None
        self.last_frame = None
        self.last_second = None
        self.next_frame_callbacks = []
        self.background_frames = []
        self.process_thread = None
        self.logger = logger
        self.total_frames = None
        self.add_next_frame_callback(self.calculate_background)
        self.image_dir = "images/unsure"
        os.makedirs(self.image_dir, exist_ok=True)

    def add_next_frame_callback(self, callback):
        """添加 next_frame 回调函数"""
        self.next_frame_callbacks.append(callback)

    def _check_seekable(self, url):
        """综合判断是否支持跳帧"""
        # 方法1：协议特征判断
        non_seek_protocols = ('rtsp://', 'rtmp://', 'udp://', 'http://', 'https://')
        if any(url.startswith(p) for p in non_seek_protocols):
            return False

        # 方法2：文件扩展名判断
        video_exts = ('mp4', 'avi', 'mkv', 'mov', 'flv')
        if '.' in url and url.split('.')[-1].lower() in video_exts:
            return True

        # 方法3：动态测试跳转能力
        try:
            current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current + 10)
            return abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - (current + 10)) < 1
        except:
            return False

    def _calculate_mean(self, frame, mean, M2, frame_count):
        """计算均值和方差"""
        frame_float = frame.astype(np.float32)
        delta = frame_float - mean
        mean += delta / (frame_count + 1)
        delta2 = frame_float - mean
        M2 += delta * delta2
        return mean, M2

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

    def start_process(self, url):
        self.url = url
        self.is_seekable = self._check_seekable(url)
        self.logger.info(f"视频源可跳帧：{self.is_seekable}")

        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"

        # 检查视频是否成功打开
        if not self.cap.isOpened():
            print(f"Failed to open video stream from {url}")
            # 回退到默认模式
            self.cap = cv2.VideoCapture(url)
            print("Fallback to default capture mode")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.is_seekable:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 检查 FPS 是否有效
        if self.fps is None or self.fps <= 0:
            print("Invalid FPS value")
            return
        self.running = True
        self.last_frame = None
        self.last_dot = None

        # 启动一个线程去运行 process_video_with_ffmpeg 函数
        self.process_thread = threading.Thread(target=self.process_video)
        self.process_thread.start()

    def stop_process(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.logger.info("视频捕获对象已释放。")

    def process_video(self):
        self.logger.info(f"视频处理线程已启动,url: {self.url}")
        second = 0
        try:
            while self.running:
                step = int(config.get_instance().get('step_second', 15))
                second += step
                if self.is_seekable:  # 本地文件模式
                    if second * self.fps > self.total_frames:
                        self.logger.info(f"已到达视频末尾，停止处理")
                        self.stop_process()
                        return
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.fps * second))
                else:  # 实时流模式
                    non_zero_pixels = 0
                    while non_zero_pixels<200:
                        ret, frame = self.cap.read()
                        if not ret:
                            if self.is_seekable:
                                self.logger.info("Failed to read frame")
                                self.stop_process()
                                return
                            else:
                                self.logger.info("视频流已结束，尝试重新连接...")
                                self.start_process(self.url)
                                return
                        if self.roi is not None:
                            x, y, w, h = self.roi
                            frame = frame[y:y + h, x:x + w]
                        if self.last_frame is None:
                            self.last_frame = frame
                            continue
                        diff = cv2.absdiff(frame, self.last_frame)
                        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                        gray_diff[0:80, :] = 0
                        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                        non_zero_pixels = cv2.countNonZero(thresh)
                    dot,conf = self.dot_cnn.predict_image(frame)
                    self.logger.info(f"检测到图像变动，当前点输：{dot} {conf:.4f} 等待{step}秒结算和下注")
                    for _ in range(int(self.fps*step)):
                        ret, frame = self.cap.read()
                        if not ret:
                            if self.is_seekable:
                                self.logger.info("Failed to read frame")
                                self.stop_process()
                                return
                            else:
                                self.logger.info("视频流已结束，尝试重新连接...")
                                self.start_process(self.url)
                                return
                ret, frame = self.cap.retrieve()
                if not ret:
                    self.logger.info("Failed to read frame")
                    if self.is_seekable:
                        self.logger.info("视频流已结束!")
                        self.stop_process()
                        return
                    else:
                        self.logger.info("视频流已结束，尝试重新连接...")
                        self.start_process(self.url)
                        return
                # end = time.time()
                #self.logger.info(f"{second}解码耗时：{(end-start)*100:.4f}ms")
                if self.roi is not None:
                    x, y, w, h = self.roi
                    frame = frame[y:y + h, x:x + w]
                second = self.next_frame(frame, second,not self.is_seekable)
                #self.logger.info(f"{second}处理耗时：{(time.time() - end) * 100:.4f}ms")
        except Exception as e:
            self.logger.error(f"处理视频时发生异常: {str(e)}")
            traceback.print_exc()
        finally:
            self.logger.info("处理线程结束。")

    def next_frame(self, frame, second,force_changed=False):
        """处理每一秒采样的帧图像"""
        dot, cf = self.dot_cnn.predict_image(frame)
        if dot == 0:
            self.logger.info(f"{second} 检测骰子在动: {dot}")
            if self.is_seekable:
                return second + random.randint(2,4)
            return second
        if cf<0.97:
            self.logger.info(f"{second}检测骰子点数:{dot} 置信度 {cf:.4f} 太小 ")
            cv2.imwrite(f"{self.image_dir}/{dot}_{second}_{cf:.4f}.jpg", frame)
            return second
        self.logger.info(f"{second}检测骰子点数:{dot} 置信度 {cf:.4f}")
        if self.last_dot is None:
            self.last_dot = dot
            self.last_frame = frame
            return second
        changed = False
        if dot != self.last_dot:
            changed = True
            self.last_second = second
            self.logger.info(f"{second}检测骰子点数变动: {self.last_dot} ==> {dot}")
        elif force_changed:
            changed = True
            self.logger.info(f"{second}检测骰子位置变动: {self.last_dot} ==> {dot}")
        elif not changed and self.last_frame is not None:
            diff = cv2.absdiff(frame, self.last_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            gray_diff[0:80, :] = 0
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            non_zero_pixels = cv2.countNonZero(thresh)
            if non_zero_pixels > 100 and (self.last_second is None or second - self.last_second > 25):
                changed = True
                self.last_second = second
                self.logger.info(f"{second}检测骰子位置变动: {self.last_dot} ==> {dot}")
        if changed:
            for callback in self.next_frame_callbacks:
                callback(frame, second, dot, changed,self.last_frame)
        self.last_frame = frame
        self.last_dot = dot
        return second

    def _recognize_dice_value(self, frame, conf=0.97):
        dot, cf = self.dot_cnn.predict_image(frame)
        if cf < conf:
            self.logger.info(f"置信度过低：{cf:.4f},{dot}")
            return None
        return dot

    def calculate_background(self, frame, second, dot, changed,last_frame):
        if self.background is not None:
            if self.is_seekable:
                return
            if changed and len(self.background_frames)<60:
                self.background_frames.append(frame)
                return
        if self.is_seekable:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames = min(200, total_frames)
            step = total_frames//num_frames
            for i in range(0, total_frames, step):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.cap.read()
                if not ret:
                    break
                if self.roi is not None:
                    x, y, w, h = self.roi
                    frame = frame[y:y + h, x:x + w]
                self.background_frames.append(frame)
                if len(self.background_frames) >= num_frames:
                    break
        else:
            if changed:
                self.background_frames.append(frame)
        size = 10
        if len(self.background_frames) >= size:
            # 计算均值和标准差
            frames = self.background_frames
            mean = np.mean(frames, axis=0).astype(np.float32)
            std_dev = np.std(frames, axis=0).astype(np.float32)
            median_frame = np.median(frames, axis=0).astype(np.uint8)
            # 背景融合策略
            background = np.where(std_dev < 100, median_frame, mean).astype(np.uint8)
            background = cv2.medianBlur(background, 5)
            self.background = background

            sum_angles = 0
            for frame in self.background_frames:
                sum_angles += self.get_angle(frame)
            avg_angle = sum_angles / len(self.background_frames)
            self.background_angle_diff = round(avg_angle - 90)
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            background_path = f"output/background_{self.background_angle_diff}_{current_time}.jpg"
            cv2.imwrite(background_path, background)
            self.logger.info(f"background saved to {background_path}")
            self.background_frames.clear()
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