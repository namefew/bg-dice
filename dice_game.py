from enum import Enum

import numpy as np


class BetType(Enum):
    # '1子', '2子', '3子', '大/小', '单/双', '大/小&单/双'
    BIG = ('大',0.95,[4,5,6])
    SMALL = ('小',0.95,[1,2,3])
    ODD = ('单',0.95,[1,3,5])
    EVEN = ('双',0.95,[2,4,6])
    ONE_DOT = ('1点',4.75,[1])
    TWO_DOT = ('2点',4.75,[2])
    THREE_DOT = ('3点',4.75,[3])
    FOUR_DOT = ('4点',4.75,[4])
    FIVE_DOT = ('5点',4.75,[5])
    SIX_DOT = ('6点',4.75,[6])
    def __init__(self, name, odds, wind_dots):
        self.display_name = name
        self.odds = odds
        self.wind_dots = wind_dots

    def get_result(self, dot):
        win = dot in self.wind_dots
        return self.odds if win else -1

    def expectation(self,prediction_dict):
        sum_exp = 0
        for i in self.wind_dots:
            sum_exp += prediction_dict.get(i) * (self.odds+1)
        return sum_exp

class DiceBet:
    def __init__(self, bet_type:BetType, bet_value=1.0):
        self.bet_type = bet_type
        self.bet_value = bet_value
        self.dot = None
    def __str__(self):
        return f"{self.bet_type.display_name}"

    def check_result(self, dot):
        self.dot = dot
        return self.result()

    def result(self):
        return self.bet_type.get_result(self.dot) * self.bet_value


class DiceGame:
    def __init__(self,logger=None):
        self.logger = logger
        self.current_bets = []
        self.total_win = 0
        self.total_bets = 0
        self.min_win = 0
        self.max_win = 0
        self.last_second = None

    def reset(self) :
        self.current_bets = []
        self.total_win = 0
        self.total_bets = 0
        self.min_win = 0
        self.max_win = 0
        self.last_second = None

    def new_bets(self, type, predict_next_dots, predict_confidences,min_exp=1.00):
        prediction_dict = dict(zip(predict_next_dots, predict_confidences))
        bets = []
        if '子' in type:
            cnt = int(type.replace('子', ''))
            for c in range(0, min(cnt, len(predict_next_dots))):
                dot = predict_next_dots[c]
                for member in list(BetType):

                    if dot in member.wind_dots and len(member.wind_dots) == 1 :
                        exp = member.expectation(prediction_dict)
                        if exp >= min_exp:
                            self.logger.info(f'{member} 期望：{exp}')
                            bets.append(DiceBet(member))
        else:
            for member in list(BetType):
                if member.display_name in type:
                    exp = member.expectation(prediction_dict)
                    if exp >= min_exp:
                        self.logger.info(f'{member.display_name} 期望：{exp}')
                        bets.append(DiceBet(member))

        return bets

    def check_bets(self, second, type, current_dot, predict_next_dots,predict_confidences,min_exp = 1.00):
        if self.last_second is None or second - self.last_second > 25:
            if len(self.current_bets) > 0:
                result = 0
                for bet in self.current_bets:
                    result += bet.check_result(current_dot)
                self.total_win += result
                if self.max_win<self.total_win:
                    self.max_win = self.total_win
                if self.min_win>self.total_win:
                    self.min_win = self.total_win
                self.logger.info(f"{second}秒 结果:{current_dot} 下注:{[str(bet) for bet in self.current_bets]} 盈利:{result:.2f} 下注量:{self.total_bets} 总盈利:{self.total_win:.2f}  最高盈利:{self.max_win:.2f} 最低盈利:{self.min_win:.2f}")
            confidence_str = np.array2string(predict_confidences, separator=', ',
                                             formatter={'float_kind': lambda x: f"{x:.4f}"})

            self.logger.info(f"{second}当前：{current_dot} 预测: {predict_next_dots} 预测置信度: {confidence_str}")

            self.current_bets = self.new_bets(type, predict_next_dots, predict_confidences,min_exp=min_exp)
            self.total_bets += sum(bet.bet_value for bet in self.current_bets)
            self.logger.info(f"{second} 下注: {[str(bet) for bet in self.current_bets]}")
            self.last_second = second
