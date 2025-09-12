import os
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

import train_resnet
import threading


class DiceOnlineVideoProcessorNew:
    def __init__(self, roi=[514, 134, 224, 224], logger=None):
        self.url = None
        self.background = None
        self.background_angle_diff = 0
        self.running = False
        self.cap = None
        self.fps = None
        self.roi = roi
        self.last_frame = None
        self.next_frame_callbacks = []
        self.background_frames = []
        self.process_thread = None
        self.logger = logger
        self.add_next_frame_callback(self.calculate_background)
        self.image_dir = "images/unsure"
        os.makedirs(self.image_dir, exist_ok=True)
        # 用于存储最近的帧，用于特征提取和下注推荐
        self.recent_frames = []
        self.max_recent_frames = 10  # 保存最近10帧

        # 背景图片文件路径
        self.background_file = "online-background.jpg"
        # 初始化时尝试加载现有背景
        self.load_existing_background()

    def load_existing_background(self):
        """加载现有的背景图片（如果在一小时内）"""
        if os.path.exists(self.background_file):
            # 检查文件修改时间
            file_mtime = os.path.getmtime(self.background_file)
            current_time = time.time()
            # 如果文件在1小时内被修改过
            if (current_time - file_mtime) <= 3600:  # 3600秒 = 1小时
                try:
                    background = cv2.imread(self.background_file)
                    if background is not None:
                        self.background = background
                        self.logger.info(f"Loaded existing background from {self.background_file}")
                        # 计算角度差
                        angle = self.get_angle(background)
                        self.background_angle_diff = round(angle - 90)
                        self.logger.info(f"Background angle diff: {self.background_angle_diff}")
                except Exception as e:
                    self.logger.error(f"Failed to load existing background: {e}")
            else:
                self.logger.info("Existing background file is older than 1 hour, will calculate new background")

    def add_next_frame_callback(self, callback):
        """添加 next_frame 回调函数"""
        if callback not in self.next_frame_callbacks:
            self.next_frame_callbacks.append(callback)

    def start_process(self, url, retry_count=0):
        max_retries = 103  # 最大重试次数
        # 在创建新 VideoCapture 前显式释放旧资源
        if self.cap is not None:
            self.cap.release()
            time.sleep(1)
            self.cap = None  # 重要！避免僵尸对象
        # 先停止现有线程
        if self.process_thread and self.process_thread.is_alive():
            self.running = False
            try:
                self.process_thread.join(timeout=2)  # 添加超时
            except RuntimeError:
                pass  # 忽略自连接错误

        # 重置关键状态2
        self.url = url
        self.running = True  # 此处重置运行状态
        self.last_frame = None  # 此处需要重置关键状态变量
        self.background_frames.clear()
        self.recent_frames.clear()  # 清空最近帧缓存

        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # 添加以下选项 ↓↓↓
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 设置超时
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # 设置超时

        # 检查视频是否成功打开
        if not self.cap.isOpened():
            print(f"Failed to open video stream from {url}")
            # 回退到默认模式
            self.cap = cv2.VideoCapture(url)
            print("Fallback to default capture mode")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 修改FPS判断逻辑
        if self.fps is None or self.fps <= 0:
            self.logger.error(f"无效FPS值，准备重试({retry_count}/{max_retries})")
            if retry_count < max_retries:
                threading.Timer(5, self.start_process, args=(url, retry_count + 1)).start()
            return
        self.running = True
        self.last_frame = None
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
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("视频流读取失败，尝试重置解码器")
                    self.cap.release()  # 显式释放
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.url)  # 重新创建
                    continue
                if self.roi is not None:
                    x, y, w, h = self.roi
                    frame = frame[y:y + h, x:x + w]
                self.last_frame = frame
        except Exception as e:
            self.logger.error(f"处理视频时发生异常: {str(e)}")
            traceback.print_exc()
            self.running = False  # 确保标记为停止
            self.logger.info("准备延迟重连...")
            threading.Timer(5, self.safe_restart).start()  # 延迟重启
        finally:
            self.logger.info("处理线程结束。")

    def safe_restart(self):
        """安全的重启方法"""
        if not self.running:
            self.start_process(self.url)

    def next_frame(self):
        frame = self.last_frame
        if frame is None:
            return None
        for callback in self.next_frame_callbacks:
            callback(frame)
        return frame.copy()

    def save_background_image(self, background):
        """保存背景图片到固定文件"""
        try:
            cv2.imwrite(self.background_file, background)
            self.logger.info(f"Background saved to {self.background_file}")
        except Exception as e:
            self.logger.error(f"Failed to save background image: {e}")

    def calculate_background(self, frame):
        if frame is None:  # 新增空帧检查
            self.logger.warning("收到空帧，跳过背景计算")
            return
        # 如果已经有背景且背景帧数小于60，则继续收集帧
        if self.background is not None:
            if len(self.background_frames) < 60:
                self.background_frames.append(frame)
                return
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
            for f in self.background_frames:
                sum_angles += self.get_angle(f)
            avg_angle = sum_angles / len(self.background_frames)
            self.background_angle_diff = round(avg_angle - 90)

            # 保存背景到固定文件
            self.save_background_image(background)

            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            background_path = f"images/background_{self.background_angle_diff}_{current_time}.jpg"
            cv2.imwrite(background_path, background)
            self.logger.info(f"Background also saved to {background_path}")
            self.background_frames.clear()

    def get_angle(self, image):
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
            # 绘制拟合的椭圆
            image_with_ellipse = image.copy()
            cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)

            return rotation_angle

        else:
            print("未检测到紫色区域的轮廓！")
            return 90
