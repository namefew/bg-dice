import asyncio
import traceback
from datetime import datetime, timedelta
import numpy as np
import cv2

from online_video_processor_new import DiceOnlineVideoProcessorNew
from ..managers.frame_manager import FrameManager


class VideoProcessorManager:
    def __init__(self, logger=None, roi=[514, 134, 224, 224]):
        self.logger = logger
        self.roi = roi
        self.processor = DiceOnlineVideoProcessorNew(roi=roi, logger=logger)
        self.frame_manager = FrameManager(buffer_duration=60)  # 60秒缓冲
        self.running = False
        
        # 添加帧回调，将帧保存到帧管理器
        self.processor.add_next_frame_callback(self._frame_callback)
        
    def _frame_callback(self, frame):
        """帧回调函数，将帧添加到帧管理器"""
        if frame is not None:
            asyncio.create_task(self.frame_manager.add_frame(frame))
            
    def start_processing(self, stream_url):
        """开始视频处理"""
        if not self.processor.running:
            self.processor.url = stream_url
            self.processor.start_process(stream_url)
            self.running = True
            
    def stop_processing(self):
        """停止视频处理"""
        self.processor.stop_process()
        self.running = False
        
    async def get_frames_in_range(self, start_time: datetime, duration: float) -> list:
        """
        获取指定时间范围内的帧
        :param start_time: 开始时间
        :param duration: 持续时间（秒）
        :return: 帧列表
        """
        end_time = start_time + timedelta(seconds=duration)
        return await self.frame_manager.get_frames_in_window(start_time, end_time)
        
    async def get_frames_around_time(self, target_time: datetime, before: float = 2.0, after: float = 2.0) -> list:
        """
        获取指定时间前后范围内的帧
        :param target_time: 目标时间
        :param before: 之前多少秒
        :param after: 之后多少秒
        :return: 帧列表
        """
        start_time = target_time - timedelta(seconds=before)
        end_time = target_time + timedelta(seconds=after)
        return await self.frame_manager.get_frames_in_window(start_time, end_time)
        
    def next_frame(self):
        """获取下一帧"""
        return self.processor.next_frame()
        
    @property
    def background(self):
        """获取背景"""
        return self.processor.background
        
    @property
    def background_history(self):
        """获取背景历史"""
        return self.processor.background_history.copy()
        
    @property
    def background_angle_diff(self):
        """获取背景角度差"""
        return self.processor.background_angle_diff
        
    def extract_frames_for_period(self, game, count=3, step=2):
        """提取指定时间段的帧"""
        return self.processor.start_extract_frame(game, count=count, step=step)
        
