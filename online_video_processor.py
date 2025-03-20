import os
import time

import cv2
import numpy as np
import train_resnet
import threading


class DiceOnlineVideoProcessor:
    def __init__(self, roi, logger=None):
        self.url = None
        self.is_seekable = True
        self.background = None
        self.running = False
        self.cap = None
        self.fps = None
        self.roi = roi
        self.dot_cnn = train_resnet.CNN()
        self.last_dot = None
        self.last_frame = None
        self.last_second = None
        self.next_frame_callbacks = []
        self.background_frames = []
        self.process_thread = None
        self.logger = logger
        self.total_frames = None
        self.add_next_frame_callback(self.calculate_background)

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
        # 启动一个线程去运行 process_video_with_ffmpeg 函数
        self.process_thread = threading.Thread(target=self.process_video)
        self.process_thread.start()

    def stop_process(self):
        self.running = False
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=5)  # 设置超时时间为5秒
            if self.process_thread.is_alive():
                self.logger.error("处理线程未能在5秒内停止，强制终止。")
            else:
                self.logger.info("处理线程已停止。")
        if self.cap is not None:
            self.cap.release()
            self.logger.info("视频捕获对象已释放。")

    def process_video(self):
        second = 0
        n = 30
        try:
            while self.running:
                start = time.time()
                second += n
                if self.is_seekable:  # 本地文件模式
                    if second * self.fps > self.total_frames:
                        self.logger.info(f"已到达视频末尾，停止处理")
                        self.stop_process()
                        return
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.fps * second))
                else:  # 实时流模式
                    for _ in range(int(self.fps * n)):
                        ret, frame = self.cap.read()
                        if not ret:
                            print("Failed to read frame")
                            self.stop_process()
                            return

                ret, frame = self.cap.retrieve()
                if not ret:
                    print("Failed to read frame")
                    self.stop_process()
                    return
                end = time.time()
                self.logger.info(f"{second}解码耗时：{(end-start)*100:.4f}ms")
                if self.roi is not None:
                    x, y, w, h = self.roi
                    frame = frame[y:y + h, x:x + w]
                second = self.next_frame(frame, second)
                self.logger.info(f"{second}处理耗时：{(time.time() - end) * 100:.4f}ms")
        except Exception as e:
            self.logger.error(f"处理视频时发生异常: {str(e)}")
        finally:
            self.logger.info("处理线程结束。")

    def next_frame(self, frame, second):
        """处理每一秒采样的帧图像"""
        dot = self._recognize_dice_value(frame, 0.99)
        if dot is None or dot == 0:
            return second-10
        if self.last_dot is None:
            self.last_dot = dot
            self.last_frame = frame
            return second
        changed = False
        if dot != self.last_dot:
            changed = True
            self.logger.info(f"Detected dice value changed: last_dot={self.last_dot}, current_dot= {dot}")
        if not changed and self.last_frame is not None:
            diff = cv2.absdiff(frame, self.last_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            gray_diff[0:80, :] = 0
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            non_zero_pixels = cv2.countNonZero(thresh)
            if non_zero_pixels > 300:
                changed = True
                self.logger.info(f"Dice movement detected: last_dot={self.last_dot}, current_dot={dot}")
        for callback in self.next_frame_callbacks:
            callback(frame, second, dot, changed)
        self.last_frame = frame
        self.last_dot = dot
        return second

    def _recognize_dice_value(self, frame, conf=0.99):
        dot, cf = self.dot_cnn.predict_image(frame)
        if cf < conf:
            return None
        return dot

    def calculate_background(self, frame, second, dot, changed):
        if self.background is not None:
            return
        if changed:
            self.background_frames.append(frame)
        size = 10 if self.is_seekable else 100
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
            cv2.imwrite(f"output/background_{second}_{dot}.jpg", background)
            self.logger.info("background calculated")
            self.background_frames.clear()
