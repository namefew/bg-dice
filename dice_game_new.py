from datetime import datetime
from enum import IntEnum


class GameStatus(IntEnum):
    """
   游戏状态枚举
   - BETTING: 投注中（玩家可下注）
   - WAITING_RESULT: 开奖中（摇骰子、等待结果）
   - HANDLE_RESULT: 结算中（计算输赢、更新数据）
   """
    BETTING = 1  # 投注中（玩家可下注）
    WAITING_RESULT = 2  # 开奖中（摇骰子、等待结果）
    HANDLE_RESULT = 3  # 结算中（计算输赢、更新数据）

class DiceGame:
    def __init__(self,round_id,seq_no,table_id='B21',table_name='极速骰宝B21'):
        self.table_id = table_id
        self.table_name = table_name
        self.round_id = round_id
        self.seq_no = seq_no
        self.start_time = datetime.now()
        self.status = GameStatus.BETTING
        self.begin_frame = None
        self.last_game_result = None

        self.recommend = None
        self.recommend_confidence = None

        self.end_time = None
        self.end_frame = None
        self.result = None

        #"chips":[{"num":7,"type":1610612737,"betTypeName":"BIG","amt":3177.255997657776},{"num":2,"type":1610612738,"betTypeName":"SMALL","amt":16.170059949159622},{"num":0,"type":1610612740,"betTypeName":"ODD","amt":0},{"num":2,"type":1610612744,"betTypeName":"EVEN","amt":180},{"num":4,"type":1610612752,"betTypeName":"DICE","amt":417.72099912166595}]
        self.chips = None
        self.frames = None
    def win(self):
        if self.recommend is not None:
            if self.recommend == self.result:
                return 4.75
            else:
                return -1
        return 0
    def to_string(self):
        return f"{self.table_name},{self.seq_no},{self.round_id},状态:{self.status},点数:{self.last_game_result},概率:{self.recommend_confidence},推荐：{self.recommend},结果:{self.result} "

    def bet_stat_message(self):
        # str = f"{self.table_name},{self.seq_no},{self.round_id},状态:{self.status},点数:{self.last_game_result}"
        str = "本局下注统计:"
        if self.chips is not None:
            split = '|'
            for chip in self.chips:
                str += f"\n{split:>26} {chip['betTypeName']:<15}{chip['num']:<3}/{round(chip['amt'],2):>10}(￥)"
        return str
    def is_status(self, status: GameStatus) -> bool:
        return self.status == status

    def update_status(self, new_status: GameStatus):
        self.status = new_status
