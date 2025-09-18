import numpy as np
import traceback
from datetime import datetime
from typing import List, Tuple, Optional

import train_dnn_torch
from video_processor import DiceVideoProcessor


class FeatureExtractor:
    def __init__(self, logger=None):
        self.logger = logger
        
    def to_input_feature(self, game, video_processor: DiceVideoProcessor = None, frame=None):
        """
        提取输入特征
        """
        if frame is None:
            frame = game.begin_frame
            
        if frame is None or video_processor is None and video_processor.background is None:
            return None
            
        if video_processor is None:
            return None
            
        # 提取视觉特征
        features = video_processor.detect_dice_feature(frame)
        if features is None:
            return None

        # 提取时间特征
        minute = game.start_time.minute
        hour = game.start_time.hour
        day = game.start_time.day
        month = game.start_time.month
        year = game.start_time.year
        week_day = game.start_time.weekday()

        # 提取游戏特征
        seq_no = int(game.seq_no) if game.seq_no is not None else -1
        last_game_result = float(game.last_game_result) if game.last_game_result is not None else -1.0
        recommend = float(game.recommend) if game.recommend is not None else -1.0
        chips = [float(chip['amt']) for chip in game.chips] if game.chips is not None else [-1, -1, -1, -1, -1]

        additional_features = [year, month, day, hour, minute, week_day, seq_no, last_game_result, recommend]
        return np.concatenate((features.flatten(), chips, additional_features))
        
    def save_all_features(self, 
                          game, 
                          input_frames: List[np.ndarray], 
                          output_frames: List[np.ndarray],
                          backgrounds: List[np.ndarray],
                          background_angle_diffs: List[float],
                          logger=None):
        """
        保存所有游戏时间序列特征
        通过组合输入帧、输出帧和背景图生成 INPUT × OUTPUT × M 个特征样本
        """
        if logger is None:
            logger = self.logger
            
        try:
            labels = [game.round_id, game.start_time.isoformat(), game.last_game_result, game.result, game.seq_no]
            
            # 遍历所有背景图
            for i, background in enumerate(backgrounds):
                angle_diff = background_angle_diffs[i] if i < len(background_angle_diffs) else 0
                video_processor = DiceVideoProcessor(background=background, logger=logger)
                video_processor.background_angle_diff = angle_diff
                
                # 遍历所有输入帧
                for input_frame in input_frames:
                    game.begin_frame = input_frame
                    input_features = self.to_input_feature(game, video_processor, input_frame)
                    if input_features is None:
                        continue
                        
                    # 遍历所有输出帧
                    for output_frame in output_frames:
                        output_features = video_processor.detect_dice_feature(output_frame)
                        if output_features is None:
                            continue
                            
                        output = np.array([
                            output_features[0],  # x
                            output_features[1],  # y
                            output_features[2],  # w
                            output_features[3],  # h
                            game.result  # next_dot
                        ])
                        
                        # 保存特征批次
                        train_dnn_torch.save_features_batch(input_features, output, labels)
                        
            # 刷新特征缓存
            train_dnn_torch.flush_features_cache()
            
        except Exception as e:
            if logger:
                logger.error(f"Error in save_all_features: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            game.frames = None