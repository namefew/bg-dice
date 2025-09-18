import asyncio
from datetime import datetime
from typing import List, Optional
from collections import deque

from dice_game import DiceGame


class GameManager:
    def __init__(self, table_id='B21', max_games=1000):
        self.table_id = table_id
        self.max_games = max_games
        self.games: List[DiceGame] = []
        self.lock = asyncio.Lock()
        # 帧缓存：存储最近60秒的帧 (假设30FPS，存储2分钟的帧)
        self.frame_buffer = deque(maxlen=3600)  # 60秒 * 60 FPS
        
    def add_frame(self, frame, timestamp=None):
        """添加帧到缓存"""
        if timestamp is None:
            timestamp = datetime.now()
        self.frame_buffer.append((timestamp, frame))
        
    def get_frames_in_window(self, start_time, end_time):
        """获取时间窗口内的帧"""
        frames = []
        for timestamp, frame in self.frame_buffer:
            if start_time <= timestamp <= end_time:
                frames.append((timestamp, frame))
        return frames

    def add_game(self, game: DiceGame):
        """添加新游戏"""
        self.games.append(game)

    def find_game(self, seq_no):
        """根据序列号查找游戏"""
        return next((game for game in self.games if game.seq_no == seq_no), None)
        
    def find_game_by_round_id(self, round_id):
        """根据局号查找游戏"""
        return next((game for game in self.games if game.round_id == round_id), None)
        
    def get_games(self):
        """获取所有游戏的副本"""
        return self.games.copy()
            
    def get_latest_game(self) -> Optional[DiceGame]:
        """获取最新的游戏"""
        if self.games:
            return self.games[-1]
        return None
        
    def recover_game_state(self, seq_no, default_state=None):
        """
        恢复游戏状态（当收到消息但没有找到游戏时）
        """
        # 检查序列号是否合理（与最新游戏相差不大）
        latest_game = self.get_latest_game()
        if latest_game and abs(int(seq_no) - int(latest_game.seq_no)) <= 5:
            # 创建一个恢复的游戏状态
            recovered_game = DiceGame(
                round_id=f"RECOVERED_{seq_no}",
                seq_no=seq_no,
                table_id=self.table_id
            )
            if default_state:
                recovered_game.status = default_state
            self.games.append(recovered_game)
            return recovered_game
        return None