import asyncio
import traceback
from datetime import datetime, time
from enum import IntEnum
from typing import List

import numpy as np
import websockets
import json
import time
import config
import dice_classifier_1
import train_resnet
from dice_classifier_big_odd import BigOddClassifier
from feature_storage import TimeSeriesFeatureStorage
from logger import LogManager
from online_video_processor_new import DiceOnlineVideoProcessorNew
from video_processor import DiceVideoProcessor


class CmdType(IntEnum):
    HALL_CMD = 1
    GAME_CMD = 2
    DEALER_CMD = 3
    BET_CMD = 4
    IM_CMD = 6
    HELP_CMD = 15
    TEMP_CMD = 5
class ActionType(IntEnum):
    NEW_ACT = 5
    BETTING_STAT_ACT = 35
    STOP_BET_ACT = 7
    RESULT_ACT = 4

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
    def win(self):
        if self.recommend is not None:
            if self.recommend == self.result:
                return 4.75
            else:
                return -1
        return 0
    def to_string(self):
        return f"{self.table_name},{self.seq_no},{self.round_id},状态:{self.status},点数:{self.last_game_result},推荐置信度:{self.recommend_confidence},推荐：{self.recommend},结果:{self.result} "

    def bet_stat_message(self):
        # str = f"{self.table_name},{self.seq_no},{self.round_id},状态:{self.status},点数:{self.last_game_result}"
        str = "本局下注统计:"
        if self.chips is not None:
            split = '|'
            for chip in self.chips:
                str += f"\n{split:>26} {chip['betTypeName']:<15}{chip['num']:<3}/{round(chip['amt'],2):>10}(￥)"
        return str

class DiceServer:

    def __init__(self, port=8765,logger=None):
        self.table_id='B21'
        self.port = port
        if logger is None:
            logger = LogManager.setup()
        self.logger = logger
        self.clients = set()
        self.lock = asyncio.Lock()
        self.sent_clients = set()  # 用于存储已发送过消息的客户端
        self.messages = {}
        self.processor =  DiceOnlineVideoProcessorNew(logger=logger)
        self.dot_cnn = train_resnet.get_cnn_instance()
        self.predict_one_cnn = dice_classifier_1.get_cnn_instance()
        self.predict_big_odd_cnn = BigOddClassifier()
        self.timeseries_storage = TimeSeriesFeatureStorage()

        self.games: List[DiceGame]= []

    def save_game_timeseries_features(self, game: DiceGame):
        """保存游戏的时间序列特征"""
        features = self.to_feature(game)
        if features is not None:
            self.timeseries_storage.save_features(features, game)
            # self.logger.info(f"Saved time series features for game {game.seq_no}")

    def to_feature(self,game:DiceGame):
        if self.processor.background is None or game.begin_frame is None:
            return None
        video_processor = DiceVideoProcessor(self.processor.background)
        features = video_processor.detect_dice_feature(game.begin_frame)
        if features is None:
            return None
        if self.processor.background is None or game.begin_frame is None:
            return None
        video_processor = DiceVideoProcessor(self.processor.background)
        features = video_processor.detect_dice_feature(game.begin_frame)
        if features is None:
            return None
        minute = game.start_time.minute
        hour = game.start_time.hour
        day = game.start_time.day
        month = game.start_time.month
        year = game.start_time.year
        week_day = game.start_time.weekday()
        seq_no = int(game.seq_no) if game.seq_no is not None else -1
        last_game_result = float(game.last_game_result) if game.last_game_result is not None else -1.0
        recommend = float(game.recommend) if game.recommend is not None else -1.0
        #result = float(game.result) if game.result is not None else -1.0
        # 拼接特征向量
        additional_features = [year, month, day, hour, minute, week_day, seq_no, last_game_result, recommend]
        return np.concatenate((features.flatten(), additional_features))

    async def handler(self, websocket):
        async with self.lock:
            self.clients.add(websocket)
            self.logger.info(f"New client connected: {websocket.remote_address}, total {len(self.clients)}")
        try:
            async for message in websocket:
                self.logger.debug(f"Received message: {message}  from:[{websocket.remote_address}]:")
                if not message.startswith('{') and not message.endswith('}'):
                    if websocket in self.clients:
                        self.clients.remove(websocket)
                        self.sent_clients.add(websocket)
                    await self.broadcast(message, exclude=[websocket])
                else:
                    await self.handle_json_message(message, websocket)
        except websockets.exceptions.ConnectionClosedError:
            pass
        except Exception as e:
            self.logger.warning(f"Error handling message from [{websocket.remote_address}]: {e}")
        finally:
            async with self.lock:
                if websocket in self.sent_clients:
                    self.sent_clients.remove(websocket)
                if websocket in self.clients:
                    self.clients.remove(websocket)
                self.logger.info(f"Client disconnected: {websocket.remote_address}")

    async def broadcast(self, message, exclude=None):
        async with self.lock:
            if not self.validate(message):
                self.logger.info(f"Duplicate message detected: {message}. Skipping broadcast.")
                return
            self.logger.info(f"Broadcasting message to {len(self.clients)} clients: {message}")
            tasks = [client.send(message) for client in self.clients if client not in (exclude or [])]
            await asyncio.gather(*tasks)
            # 改为保留最近10000条记录
            if len(self.messages) > 10000:
                oldest_key = next(iter(self.messages))
                del self.messages[oldest_key]

    # 在WebSocketServer类中添加以下方法
    async def udp_listener(self):
        class UDPProtocol:
            def __init__(self, server):
                self.server = server

            def connection_made(self, transport):
                self.transport = transport

            def datagram_received(self, data, addr):
                message = data.decode()
                self.logger.debug(f"Received UDP message from {addr}: {message}")
                # 创建任务异步处理消息
                asyncio.create_task(self.server.broadcast(message))

        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self),
            local_addr=('0.0.0.0', 5005)
        )
        self.logger.info(f"UDP listener started on port 5005")

    # 修改start方法
    async def start(self):
        async with websockets.serve(self.handler, "0.0.0.0", self.port):
            self.logger.info(f"WebSocket server started on port {self.port}")
            # 启动UDP监听
            await self.udp_listener()
            await asyncio.Future()  # 永久运行

    def validate(self, message):

        key, timestamp = message.rsplit(',', 1)  # 从右侧分割一次
        timestamp = float(timestamp)
        if key in self.messages:
            if abs(timestamp - self.messages[key])< 600:  # 600秒内重复消息将被丢弃
                return False
            else:
                self.messages[key] = timestamp
                return True
        else:
            self.messages[key] = timestamp
            return True
    def find_game(self, seq_no):
        return next((game for game in self.games if game.seq_no == seq_no), None)

    async def handle_json_message(self, message, websocket):
        try:
            # 解析JSON消息
            data = json.loads(message)
            if 'cmd' in data and 'action' in data and data['cmd'] == CmdType.GAME_CMD:
                if data['action'] == ActionType.NEW_ACT:
                    await self.handle_new_game(data)
                elif data['action'] == ActionType.BETTING_STAT_ACT:
                    await self.handle_bet_statement(data)
                elif data['action'] == ActionType.STOP_BET_ACT:
                    await self.handle_game_stop_betting(data)
                elif data['action'] == ActionType.RESULT_ACT:
                    await self.handle_game_result(data)

            elif 'streamUrl' in data:
                stream_url = data['streamUrl']
                self.processor.url = stream_url
                if not self.processor.running:
                    self.processor.start_process(stream_url)

        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON message: {message}")
        except Exception as e:
            self.logger.error(f"Error handling JSON message: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def handle_game_stop_betting(self, data):
        # {"cmd":2,"cmdName":"GAME_CMD","action":7,"actionName":"STOP_BET_ACT","size":12,"ext":0,"body":{"tableID":"B21","seqNo":1761979}}
        if data['body']['tableID'] == self.table_id:
            oldGame = self.find_game(data['body']['seqNo'])
            if oldGame is not None:
                if oldGame.status == GameStatus.WAITING_RESULT:
                    return None
                oldGame.status = GameStatus.WAITING_RESULT
                self.logger.info(f"{oldGame.bet_stat_message()} ")
                self.logger.info(f"{oldGame.to_string()} 停止下注，开奖中...")
            else:
                self.logger.warning(f"未找到游戏：{data['body']['seqNo']}")

    async def handle_new_game(self, data):
        # {"cmd":2,"cmdName":"GAME_CMD","action":5,"actionName":"NEW_ACT","size":28,"ext":25,"body":{"gameDto":{"tableID":"B21","seqNo":1761980},"serialNo":"BGB212509116E9"}}
        body = data['body']
        tableID = body['gameDto']['tableID']
        seqNo = body['gameDto']['seqNo']
        serialNo = body['serialNo']
        if tableID != self.table_id:
            return None
        oldGame = self.find_game(seqNo)
        if oldGame is None:
            game = DiceGame(round_id=serialNo, seq_no=seqNo, table_id=tableID)
            self.games.append(game)
            # 异步执行耗时操作
            loop = asyncio.get_running_loop()
            last_game = self.find_game(seqNo - 1)
            if last_game is not None:
                game.last_game_result = last_game.result
                game.begin_frame = last_game.end_frame
            if game.begin_frame is None:
                frame = await loop.run_in_executor(None, self.processor.next_frame)
                if frame is None:
                    self.logger.info(f"{game.to_string()} 视频还未开始")
                    return None
                game.begin_frame = frame
            if game.last_game_result is None:
                # 异步执行CNN预测
                dot, cf = await loop.run_in_executor(None, self.dot_cnn.predict_image, game.begin_frame)
                game.last_game_result = dot
            if self.processor.background is None:
                self.logger.info(f"{game.to_string()} 背景图片计算中……")
                return None
            # 异步执行推荐预测
            predict_dots, confidences = await loop.run_in_executor(
                None,
                self.predict_one_cnn.predict_image_top,
                game.begin_frame,
                self.processor.background,
                self.processor.background_angle_diff
            )
            if len(predict_dots) > 0:
                game.recommend = int(predict_dots[0])
                game.recommend_confidence = round(confidences[0], 4)
                self.logger.info(f"{game.to_string()} 推荐: {game.recommend} {game.recommend_confidence}")
                recommend_gate = config.get_instance().get('single_min_rate', 0.2)
                if game.recommend_confidence >= recommend_gate:
                    # 发送推荐
                    broadcast_msg = f"{game.table_id},{game.seq_no},{game.recommend},{game.recommend_confidence},{time.time()}"
                    self.logger.info(f"发送广播: {broadcast_msg} ...")
                    await self.broadcast(broadcast_msg)
                else:
                    self.logger.info(
                        f"{game.to_string()} 推荐失败: {game.recommend} {game.recommend_confidence} < 阈值:{recommend_gate}")
            else:
                self.logger.info(f"{game.to_string()} 推荐失败: 预测结果为空")
        else:
            self.logger.info(f"游戏已开始{self.table_id}-{seqNo} {oldGame}")

    async def handle_game_result(self, data):
        # {"cmd":2,"cmdName":"GAME_CMD","action":4,"actionName":"RESULT_ACT","size":20,"ext":6,"body":{"tableID":"B21","seqNo":1761979,"position":1,"count":1,"padding":0,"result":[4]}}
        if data['body']['tableID'] != self.table_id:
            return None
        game = self.find_game(data['body']['seqNo'])
        if game is not None:
            if game.status == GameStatus.HANDLE_RESULT:
                return None
            game.status = GameStatus.HANDLE_RESULT
            game.result = data['body']['result'][0]
            game.end_frame = self.processor.next_frame()
            game.end_time = time.time()
            self.logger.info(f"{game.to_string()}  本局盈利：{game.win()}")
            self.logger.info(self.statement())

            # 强制添加样本
            if self.predict_one_cnn.is_force_add_sample():
                self.predict_one_cnn.add_sample(current_dot=game.result, last_frame=game.begin_frame,
                                               background=self.processor.background,
                                               angle_diff=self.processor.background_angle_diff)
            # 保存特征用于后续分析
            self.save_game_timeseries_features(game)

        else:
            self.logger.warning(f"未找到游戏：{data['body']['seqNo']}")

    async def handle_bet_statement(self, data):
        #{"cmd":2,"cmdName":"GAME_CMD","action":35,"actionName":"BETTING_STAT_ACT","size":108,"ext":0,"body":{"type":6,"gameTypeName":"SPEED_SICBO","tableID":"B21","count":5,"playerTotal":0,"chipsTotal":0,"chips":[{"num":7,"type":1610612737,"betTypeName":"BIG","amt":3177.255997657776},{"num":2,"type":1610612738,"betTypeName":"SMALL","amt":16.170059949159622},{"num":0,"type":1610612740,"betTypeName":"ODD","amt":0},{"num":2,"type":1610612744,"betTypeName":"EVEN","amt":180},{"num":4,"type":1610612752,"betTypeName":"DICE","amt":417.72099912166595}]}}
        if data['body']['tableID'] != self.table_id:
            return None
        if not self.games:
            return None
        game = self.games[-1]
        if game is not None:
            chips = data['body']['chips']
            game.chips = chips
        else:
            self.logger.warning(f"未找到游戏：{data['body']['seqNo']}")

    def statement(self):
        all = 0
        max = 0
        min = 0
        win = 0
        for game in self.games:
            if game.recommend is not None:
                all+=1
                win += game.win()
                if max<win:
                    max=win
                if min>win:
                    min=win
        win_rate = 0
        if all!= 0:
            win_rate = round(win/all*100,2)
        return f"盈利率：{win_rate}% 下注总量:{all} 最大盈利:{max:.2f} 最小盈利:{min:.2f} 当前盈利:{win:.2f}"



# 启动WebSocket服务
if __name__ == "__main__":
    server = DiceServer(port=config.get_instance().get('server_port', 8765))
    asyncio.run(server.start())