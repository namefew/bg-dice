import asyncio
import traceback
from datetime import datetime
from enum import IntEnum
import concurrent.futures
from typing import List
import os
import numpy as np
import websockets
import json
import time
import config
import dice_classifier_1
import train_dnn_torch
import train_resnet
from dice_classifier_big_odd import BigOddClassifier
from dice_game_new import DiceGame, GameStatus
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

class DiceServer:
    def __init__(self, port=8765, logger=None):
        self.table_id = 'B21'
        self.port = port
        if logger is None:
            logger = LogManager.setup()
        self.logger = logger
        self.clients = set()
        self.lock = asyncio.Lock()
        self.sent_clients = set()
        self.messages = {}

        # 创建专用线程池

        # 根据CPU核心数设置线程池大小，通常设置为CPU核心数+1或2*CPU核心数
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=(os.cpu_count() or 1) + 4,
            thread_name_prefix="dice_processor"
        )

        self.dot_cnn = train_resnet.get_cnn_instance()
        self.predict_one_cnn = dice_classifier_1.get_cnn_instance()
        self.predict_big_odd_cnn = BigOddClassifier()
        self.timeseries_storage = TimeSeriesFeatureStorage()
        self.processor = DiceOnlineVideoProcessorNew(logger=logger)
        self.games: List[DiceGame] = []

    async def close(self):
        """显式关闭资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)

    def save_game_timeseries_features(self, game: DiceGame):
        """保存游戏的时间序列特征"""
        features = self.to_input_feature(game)
        if features is not None:
            self.timeseries_storage.save_features(features, game)
            # self.logger.info(f"Saved time series features for game {game.seq_no}")

    def save_all_features(self, game:DiceGame, input_frames, output_frames):
        """保存所有游戏时间序列特征"""
        try:
            labels = [game.round_id,game.start_time.strftime("%Y-%m-%d %H:%M:%S"),game.last_game_result,game.result,game.seq_no]
            video_processor = DiceVideoProcessor(background=self.processor.background, logger=self.logger)
            backgrounds = self.processor.background_history[-3:] if len(
                self.processor.background_history) >3 else self.processor.background_history.copy()
            if self.processor.background is not None:
                backgrounds.append(self.processor.background)
            self.logger.info(f"{game.seq_no}局保存特征 {len(input_frames)} X {len(output_frames)} X {len( backgrounds)} ...")
            for background in backgrounds:
                video_processor.background = background
                for i in range(len(input_frames)):
                    game.begin_frame = input_frames[i]
                    input_features = self.to_input_feature(game,video_processor)
                    if input_features is None:
                        continue
                    for j in range(len(output_frames)):
                        output_features = video_processor.detect_dice_feature(output_frames[j])
                        if output_features is None:
                            continue
                        output = np.array([
                            output_features[0],  # x
                            output_features[1],  # y
                            output_features[2],  # w
                            output_features[3],  # h
                            game.result  # next_dot
                        ])
                        train_dnn_torch.save_features_batch(input_features, output, labels)
            train_dnn_torch.flush_features_cache()
            self.logger.info(f"{game.seq_no}局保存特征完成,共{len(input_frames) * len(output_frames) * len(backgrounds)} 个")
        except Exception as e:
            self.logger.error(f"Error in save_all_features: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            game.frames = None

    def to_input_feature(self, game: DiceGame, video_processor: DiceVideoProcessor = None):
        if game.begin_frame is None or video_processor is None and self.processor.background is None:
            return None
        if video_processor is None:
            video_processor = DiceVideoProcessor(background=self.processor.background, logger=self.logger)
        # 提取视觉特征
        features = video_processor.detect_dice_feature(game.begin_frame)
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
        chips = [float(chip['amt']) for chip in game.chips] if game.chips is not None else [-1,-1,-1,-1,-1]

        additional_features = [year, month, day, hour, minute, week_day, seq_no, last_game_result, recommend]
        return np.concatenate((features.flatten(), chips,additional_features))
        
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
        udp_port = config.get_instance().get('udp_port', 5005)
        await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self),
            local_addr=('0.0.0.0', udp_port)
        )
        self.logger.info(f"UDP listener started on port {udp_port}")

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
                if oldGame.is_status(GameStatus.WAITING_RESULT):
                    return None
                oldGame.update_status(GameStatus.WAITING_RESULT)
                self.logger.info(f"{oldGame.bet_stat_message()} ")
                self.logger.info(f"{oldGame.to_string()} 停止下注，开奖中...")
            else:
                self.logger.warning(f"未找到游戏：{data['body']['seqNo']} 状态{GameStatus.WAITING_RESULT}")

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
                frame = await loop.run_in_executor(self.thread_pool, self.processor.next_frame)
                if frame is None:
                    self.logger.warning(f"{game.to_string()} 视频还未开始")
                    return None
                game.begin_frame = frame
            future = self.processor.start_extract_frame(game.seq_no, count=6, step=2)
            future.add_done_callback(lambda f: self._frame_extract_callback(game, f))
            if game.last_game_result is None:
                dot, cf = await loop.run_in_executor(self.thread_pool, self.dot_cnn.predict_image, game.begin_frame)
                game.last_game_result = dot

            if self.processor.background is None:
                self.logger.info(f"{game.to_string()} 背景图片计算中……")
                return None
            # 异步执行推荐预测
            predict_dots, confidences = await loop.run_in_executor(
                self.thread_pool,
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
            # 将帧提取也放到线程池中执行，避免阻塞事件循环

        else:
            self.logger.info(f"重复的消息{self.table_id}-{seqNo} {oldGame.to_string()}")

    def _frame_extract_callback(self, game, future):
        """帧提取完成的回调函数"""
        try:
            game.frames = future.result()
            self.logger.info(f"{game.seq_no}局 帧数：{len(game.frames)} 提取完成")
        except Exception as e:
            self.logger.error(f"帧提取回调出错: {e}")

    async def handle_game_result(self, data):
        # {"cmd":2,"cmdName":"GAME_CMD","action":4,"actionName":"RESULT_ACT","size":20,"ext":6,"body":{"tableID":"B21","seqNo":1761979,"position":1,"count":1,"padding":0,"result":[4]}}
        if data['body']['tableID'] != self.table_id:
            return None
        game = self.find_game(data['body']['seqNo'])
        if game is not None:
            if game.is_status(GameStatus.HANDLE_RESULT):
                return None
            game.update_status(GameStatus.HANDLE_RESULT)
            game.result = data['body']['result'][0]
            game.end_frame = self.processor.next_frame()
            game.end_time = time.time()
            self.logger.info(f"{game.to_string()}  本局盈利：{game.win()}")
            self.logger.info(await self.statement())
            # 异步执行样本添加（不需要等待帧提取完成）
            asyncio.create_task(
                self._run_with_error_handling(
                    self.predict_one_cnn.add_sample,
                    game.result,  # current_dot
                    game.begin_frame,  # last_frame
                    self.processor.background,  # background
                    self.processor.background_angle_diff,  # angle_diff
                    task_name="add_sample"
                )
            )

            # 异步执行特征保存
            asyncio.create_task(
                self._run_with_error_handling(
                    self.save_game_timeseries_features,
                    game,
                    task_name="save_game_timeseries_features"
                )
            )

            if game.frames is not None and len(game.frames) > 0:
                # 异步执行帧提取和特征保存
                end_frame_future = self.processor.start_extract_frame(game, count=6, step=2)
                end_frame_future.add_done_callback(lambda f: self._save_all_features_callback(game, f))
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
            if game.begin_frame is None:
                game.begin_frame = self.processor.next_frame()
                game.start_time = datetime.now()

    async def _run_with_error_handling(self, func, *args, task_name="Task"):
        """带错误处理的线程池任务执行"""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.thread_pool, func, *args)
        except Exception as e:
            self.logger.error(f"Error in {task_name}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    def _save_all_features_callback(self,game,future):
        end_frames = future.result()
        start_frames = game.frames
        if start_frames is not None and len(start_frames) > 0 and end_frames is not None and len(end_frames) > 0:
            self.logger.info(f"{game.seq_no} 局开始保存特征...")
            self.save_all_features(
                game,
                start_frames,
                end_frames
            )

    async def statement(self):
        all = 0
        max_val = 0
        min_val = 0
        win = 0
        games_to_check = self.games.copy()
        for game in games_to_check:
            if game.recommend is not None:
                all += 1
                win += game.win()
                if max_val < win:
                    max_val = win
                if min_val > win:
                    min_val = win
        win_rate = 0
        if all != 0:
            win_rate = round(win / all * 100, 2)
        return f"盈利率：{win_rate}% 下注总量:{all} 最大盈利:{max_val:.2f} 最小盈利:{min_val:.2f} 当前盈利:{win:.2f}"

#启动WebSocket服务
if __name__ == "__main__":
    config.start_file_watcher()
    server = DiceServer(port=config.get_instance().get('server_port', 8765))
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Server stopped by user")