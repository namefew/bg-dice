import asyncio
import os
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

import train_resnet
import threading
from collections import deque


class DiceOnlineVideoProcessorNew:
    def __init__(self, roi=[514, 134, 224, 224], logger=None):
        self.url = None
        self.background = None
        self.background_history = []
        self.background_angle_diff = 0
        self.running = False
        self.cap = None
        self.fps = None
        self.roi = roi
        self.last_frame = None
        self.next_frame_callbacks = []
        self.background_frames = []
        self.process_thread = None
        self.background_thread = None  # 用于背景计算的专用线程
        self.logger = logger
        self.add_next_frame_callback(self.calculate_background)
        self.image_dir = "images/unsure"
        os.makedirs(self.image_dir, exist_ok=True)

        # 背景图片文件路径
        self.background_file = "online-background.jpg"
        # 初始化时尝试加载现有背景
        self.load_existing_background()

        # 用于管理多个帧提取任务
        self.frame_count = 0
        self.extraction_tasks = {}  # 使用字典存储任务
        self.task_counter = 0  # 任务ID计数器

        # 帧缓存：存储最近的帧，按时间排序
        self.frame_buffer = deque(maxlen=3600)  # 默认存储60秒的帧（假设60FPS）

    def start_extract_frame(self, seq_no, count=3, step=3):
        # 为每个任务生成唯一ID
        self.task_counter += 1
        task_id = self.task_counter

        # 创建新的 Future 对象
        future = asyncio.Future()

        # 创建任务数据
        task_data = {
            'task_id': task_id,
            'seq_no': seq_no,
            'count': count,
            'step': step,
            'future': future,
            'recent_frames': [],
            'start_frame': self.frame_count,
            'first_frame_saved': False
        }

        # 添加到任务字典
        self.extraction_tasks[task_id] = task_data

        self.logger.debug(f"Started extraction task {task_id} for round {seq_no}, count={count}, step={step}")
        return future


    def complete_task(self, task_id):
        """完成指定任务"""
        if task_id in self.extraction_tasks:
            task_data = self.extraction_tasks[task_id]

            # 异步完成 future
            if task_data['future'] and not task_data['future'].done():
                def set_result():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        frames = task_data['recent_frames'].copy()
                        task_data['future'].set_result(frames)
                    except Exception as e:
                        if task_data['future'] and not task_data['future'].done():
                            task_data['future'].set_exception(e)

                threading.Thread(target=set_result, daemon=True).start()
            # 从任务字典中移除
            del self.extraction_tasks[task_id]
            self.logger.debug(f"Completed extraction task {task_id} for round {task_data['seq_no']}")

    def cancel_all_tasks(self):
        """取消所有未完成的任务"""
        for task_id, task_data in list(self.extraction_tasks.items()):
            if task_data['future'] and not task_data['future'].done():
                task_data['future'].cancel()
            del self.extraction_tasks[task_id]
        self.logger.info("Cancelled all extraction tasks")
        
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
        self.cancel_all_tasks()
        if self.cap is not None:
            self.cap.release()
            self.logger.info("视频捕获对象已释放。")

    def process_video(self):
        self.logger.info(f"视频处理线程已启动,url: {self.url}")
        self.frame_count = 0
        first_frame_for_round_saved = False  # 为每轮游戏保存第一帧
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
                    if (not first_frame_for_round_saved):
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            first_frame_path = f"images/{timestamp}.jpg"
                            cv2.imwrite(first_frame_path, frame)
                            self.logger.info(f"视频第一帧截图已保存到: {first_frame_path}")
                            first_frame_for_round_saved = True
                        except Exception as e:
                            self.logger.error(f"保存视频第一帧截图失败: {e}")
                    x, y, w, h = self.roi
                    frame = frame[y:y + h, x:x + w]
                self.last_frame = frame
                self.frame_count += 1

                # 将帧添加到缓存
                self.frame_buffer.append((datetime.now(), frame.copy()))

                # 当开始提取帧时保存第一帧
                # 处理所有提取任务
                completed_tasks = []
                for task_id, task_data in self.extraction_tasks.items():
                    # 提取帧
                    if (len(task_data['recent_frames']) < task_data['count'] and
                            (self.frame_count - task_data['start_frame']) % int(self.fps * task_data['step']) == 0):
                        task_data['recent_frames'].append(frame.copy())
                        # 检查任务是否完成
                        if len(task_data['recent_frames']) >= task_data['count']:
                            completed_tasks.append(task_id)

                # 完成已完成的任务
                for task_id in completed_tasks:
                    self.complete_task(task_id)
        except Exception as e:
            self.logger.error(f"处理视频时发生异常: {str(e)}")
            traceback.print_exc()
            self.running = False  # 确保标记为停止
            # 异步设置异常
            if self.frame_collection_future and not self.frame_collection_future.done():
                def set_exception():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        self.frame_collection_future.set_exception(e)
                    except:
                        pass
                threading.Thread(target=set_exception, daemon=True).start()

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
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            background_path = f"images/background_{self.background_angle_diff}_{current_time}.jpg"
            cv2.imwrite(background_path, background)
            self.logger.info(f"Background also saved to {background_path}")
        except Exception as e:
            self.logger.error(f"Failed to save background image: {e}")

    def calculate_background(self, frame):
        if frame is None:  # 新增空帧检查
            self.logger.warning("收到空帧，跳过背景计算")
            return
        # 如果已经有背景且背景帧数小于60，则继续收集帧
        if self.background is not None:
            if len(self.background_frames) < 30:
                self.background_frames.append(frame)
                return
        self.background_frames.append(frame)
        size = 10
        if len(self.background_frames) >= size:
            # 启动新线程进行背景计算
            if self.background_thread is None or not self.background_thread.is_alive():
                self.background_thread = threading.Thread(target=self._do_extract_background)
                self.background_thread.start()
            else:
                self.logger.info("背景计算线程正在运行中...")

    def _do_extract_background(self):
        """在独立线程中执行背景提取"""
        try:
            self.logger.info("开始在独立线程中计算背景...")
            # 计算均值和标准差
            frames = self.background_frames
            mean = np.mean(frames, axis=0).astype(np.float32)
            std_dev = np.std(frames, axis=0).astype(np.float32)
            median_frame = np.median(frames, axis=0).astype(np.uint8)
            # 背景融合策略
            background = np.where(std_dev < 100, median_frame, mean).astype(np.uint8)
            background = cv2.medianBlur(background, 5)
            if self.background is not None:
                self.background_history.append(self.background)
                if len(self.background_history)>10:
                    self.background_history.pop(0)
            self.background = background
            sum_angles = 0
            for f in self.background_frames:
                sum_angles += self.get_angle(f)
            avg_angle = sum_angles / len(self.background_frames)
            self.background_angle_diff = round(avg_angle - 90)

            # 保存背景到固定文件
            self.save_background_image(background)

            self.background_frames.clear()
            self.logger.info("背景计算完成")
        except Exception as e:
            self.logger.error(f"背景计算出错: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

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

    def get_frames_in_time_range(self, start_time, end_time):
        """
        获取指定时间范围内的帧
        :param start_time: 开始时间
        :param end_time: 结束时间
        :return: 帧列表
        """
        frames = []
        for timestamp, frame in self.frame_buffer:
            if start_time <= timestamp <= end_time:
                frames.append((timestamp, frame))
        return frames