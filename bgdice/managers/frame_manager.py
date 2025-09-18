import asyncio
from datetime import datetime, timedelta
from collections import deque
import numpy as np


class FrameManager:
    def __init__(self, buffer_duration=60):
        """
        初始化帧管理器
        :param buffer_duration: 缓冲区时长（秒），默认60秒
        """
        self.buffer_duration = buffer_duration
        # 使用双端队列存储帧和时间戳，自动限制大小
        self.frame_buffer = deque()
        self.lock = asyncio.Lock()
        
    async def add_frame(self, frame, timestamp=None):
        """
        添加帧到缓存
        :param frame: 视频帧
        :param timestamp: 时间戳，默认为当前时间
        """
        async with self.lock:
            if timestamp is None:
                timestamp = datetime.now()
                
            # 添加新帧
            self.frame_buffer.append((timestamp, frame.copy()))
            
            # 清理过期帧
            self._cleanup_expired_frames()
    
    def _cleanup_expired_frames(self):
        """清理过期帧"""
        if not self.frame_buffer:
            return
            
        current_time = datetime.now()
        expiry_time = current_time - timedelta(seconds=self.buffer_duration)
        
        # 移除过期帧
        while self.frame_buffer and self.frame_buffer[0][0] < expiry_time:
            self.frame_buffer.popleft()
            
    async def get_frames_in_window(self, start_time, end_time):
        """
        获取指定时间窗口内的帧
        :param start_time: 开始时间
        :param end_time: 结束时间
        :return: 时间窗口内的帧列表
        """
        async with self.lock:
            frames = []
            for timestamp, frame in self.frame_buffer:
                if start_time <= timestamp <= end_time:
                    frames.append((timestamp, frame))
            return frames
            
    async def get_frames_before(self, timestamp, count=1):
        """
        获取指定时间之前的帧
        :param timestamp: 时间戳
        :param count: 获取帧的数量
        :return: 帧列表
        """
        async with self.lock:
            frames = []
            for i in range(len(self.frame_buffer) - 1, -1, -1):
                frame_time, frame = self.frame_buffer[i]
                if frame_time <= timestamp:
                    frames.append((frame_time, frame))
                    if len(frames) >= count:
                        break
            return list(reversed(frames))
            
    async def get_recent_frames(self, count=1):
        """
        获取最近的帧
        :param count: 获取帧的数量
        :return: 最近的帧列表
        """
        async with self.lock:
            frames = []
            for i in range(len(self.frame_buffer) - 1, max(-1, len(self.frame_buffer) - count - 1), -1):
                if i >= 0:
                    frames.append(self.frame_buffer[i])
            return list(reversed(frames))
            
    async def clear_buffer(self):
        """清空帧缓冲区"""
        async with self.lock:
            self.frame_buffer.clear()