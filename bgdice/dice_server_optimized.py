import asyncio
import traceback
from enum import IntEnum
import os
import websockets
import json
import time as time_module
import config

import dice_classifier_1
import train_resnet
from dice_classifier_big_odd import BigOddClassifier
from feature_storage import TimeSeriesFeatureStorage
from logger import LogManager

from .managers.game_manager import GameManager
from .processors.video_processor_manager import VideoProcessorManager
from .extractors.feature_extractor import FeatureExtractor
from .models.dice_game_model import DiceGame, GameStatus


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


class DiceServerOptimized:
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
        import concurrent.futures
        # 根据CPU核心数设置线程池大小，通常设置为CPU核心数+1或2*CPU核心数
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=(os.cpu_count() or 1) + 2,
            thread_name_prefix="dice_processor"
        )

        self.dot_cnn = train_resnet.get_cnn_instance()
        self.predict_one_cnn = dice_classifier_1.get_cnn_instance()
        self.predict_big_odd_cnn = BigOddClassifier()
        self.timeseries_storage = TimeSeriesFeatureStorage()
        
        # 初始化管理器
        self.game_manager = GameManager(table_id=self.table_id)
        self.video_manager = VideoProcessorManager(logger=logger)
        self.feature_extractor = FeatureExtractor(logger=logger)
        
        # 游戏状态恢复相关
        self.pending_recovery = {}  # 存储待恢复的游戏状态

    async def close(self):
        """显式关闭资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        self.video_manager.stop_processing()

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
            if abs(timestamp - self.messages[key]) < 600:  # 600秒内重复消息将被丢弃
                return False
            else:
                self.messages[key] = timestamp
                return True
        else:
            self.messages[key] = timestamp
            return True

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
                self.video_manager.start_processing(stream_url)

        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON message: {message}")
        except Exception as e:
            self.logger.error(f"Error handling JSON message: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def handle_game_stop_betting(self, data):
        # {"cmd":2,"cmdName":"GAME_CMD","action":7,"actionName":"STOP_BET_ACT","size":12,"ext":0,"body":{"tableID":"B21","seqNo":1761979}}
        if data['body']['tableID'] == self.table_id:
            game = self.game_manager.find_game(data['body']['seqNo'])
            if game is not None:
                if game.is_status(GameStatus.WAITING_RESULT):
                    return None
                game.update_status(GameStatus.WAITING_RESULT)
                self.logger.info(f"{game.bet_stat_message()} ")
                self.logger.info(f"{game.to_string()} 停止下注，开奖中...")
            else:
                self.logger.warning(f"未找到游戏：{data['body']['seqNo']} 状态{GameStatus.WAITING_RESULT}")
                # 尝试恢复游戏状态
                recovered_game = self.game_manager.recover_game_state(
                    data['body']['seqNo'], 
                    default_state=GameStatus.WAITING_RESULT
                )
                if recovered_game:
                    self.logger.info(f"已恢复游戏状态: {recovered_game.to_string()}")
                else:
                    # 记录待处理消息，等待游戏创建
                    self.pending_recovery[data['body']['seqNo']] = {
                        'type': 'stop_betting',
                        'data': data,
                        'timestamp': time_module.time()
                    }

    async def handle_new_game(self, data):
        # {"cmd":2,"cmdName":"GAME_CMD","action":5,"actionName":"NEW_ACT","size":28,"ext":25,"body":{"gameDto":{"tableID":"B21","seqNo":1761980},"serialNo":"BGB212509116E9"}}
        body = data['body']
        tableID = body['gameDto']['tableID']
        seqNo = body['gameDto']['seqNo']
        serialNo = body['serialNo']
        if tableID != self.table_id:
            return None
            
        oldGame = self.game_manager.find_game(seqNo)
        if oldGame is None:
            game = DiceGame(round_id=serialNo, seq_no=seqNo, table_id=tableID)
            self.game_manager.add_game(game)

            # 异步执行耗时操作
            last_game = self.game_manager.find_game(seqNo - 1)
            if last_game is not None:
                game.last_game_result = last_game.result
                game.begin_frame = last_game.end_frame
            if game.begin_frame is None:
                frame = self.video_manager.next_frame()
                if frame is None:
                    self.logger.info(f"{game.to_string()} 视频还未开始")
                    return None
                game.begin_frame = frame
            if game.last_game_result is None:
                dot, cf =self.dot_cnn.predict_image(game.begin_frame)
                game.last_game_result = dot

            if self.video_manager.background is None:
                self.logger.info(f"{game.to_string()} 背景图片计算中……")
                return None
                
            # 异步执行推荐预测
            predict_dots, confidences = self.predict_one_cnn.predict_image_top(                game.begin_frame,
                self.video_manager.background,
                self.video_manager.background_angle_diff
            )
            if len(predict_dots) > 0:
                game.recommend = int(predict_dots[0])
                game.recommend_confidence = round(confidences[0], 4)
                self.logger.info(f"{game.to_string()} 推荐: {game.recommend} {game.recommend_confidence}")
                recommend_gate = config.get_instance().get('single_min_rate', 0.2)
                if game.recommend_confidence >= recommend_gate:
                    # 发送推荐
                    broadcast_msg = f"{game.table_id},{game.seq_no},{game.recommend},{game.recommend_confidence},{time_module.time()}"
                    self.logger.info(f"发送广播: {broadcast_msg} ...")
                    await self.broadcast(broadcast_msg)
                else:
                    self.logger.info(
                        f"{game.to_string()} 推荐失败: {game.recommend} {game.recommend_confidence} < 阈值:{recommend_gate}")
            else:
                self.logger.info(f"{game.to_string()} 推荐失败: 预测结果为空")
            
            # 使用回调函数方式设置帧提取，不等待结果
            future = self.processor.start_extract_frame(game.seq_no, count=6, step=2)
            future.add_done_callback(lambda f: self._frame_extract_callback(game, f))

        else:
            self.logger.info(f"游戏已开始{self.table_id}-{seqNo} {oldGame.to_string()}")
            
    def _frame_extract_callback(self, game, future):
        """帧提取完成的回调函数"""
        try:
            game.frames = future.result()
        except Exception as e:
            self.logger.error(f"帧提取回调出错: {e}")
            game.frames = []

    async def _extract_frames_background(self, game):
        """
        在后台异步提取帧，避免阻塞主消息处理流程
        """
        try:
            # 提取帧但不等待结果
            future = self.video_manager.extract_frames_for_period(game.seq_no, count=3, step=2)
            # 保存future引用到game对象，供后续使用
            game._frames_future = future
            
            # 在线程中等待future完成，避免阻塞主协程
            if future is not None:
                loop = asyncio.get_running_loop()
                game.frames = await loop.run_in_executor(None, self._wait_for_future, future)
            else:
                game.frames = []
                
            self.logger.info(f"为游戏 {game.seq_no} 提取帧完成，共 {len(game.frames)} 帧")
        except Exception as e:
            self.logger.error(f"提取帧时出错: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            game.frames = []
            
    def _wait_for_future(self, future):
        """在线程中等待future完成"""
        try:
            return future.result() if future is not None else []
        except Exception as e:
            print(f"等待future时出错: {e}")
            return []
            
    async def handle_game_result(self, data):
        # {"cmd":2,"cmdName":"GAME_CMD","action":4,"actionName":"RESULT_ACT","size":20,"ext":6,"body":{"tableID":"B21","seqNo":1761979,"position":1,"count":1,"padding":0,"result":[4]}}
        if data['body']['tableID'] != self.table_id:
            return None
        game = self.game_manager.find_game(data['body']['seqNo'])
        if game is not None:
            if game.is_status(GameStatus.HANDLE_RESULT):
                return None
            game.update_status(GameStatus.HANDLE_RESULT)
            game.result = data['body']['result'][0]
            game.end_frame = self.video_manager.next_frame()
            game.end_time = time_module.time()
            self.logger.info(f"{game.to_string()}  本局盈利：{game.win()}")
            self.logger.info(await self.statement())
            
            # 异步执行样本添加（不需要等待帧提取完成）
            asyncio.create_task(
                self._run_with_error_handling(
                    self.predict_one_cnn.add_sample,
                    game.result,  # current_dot
                    game.begin_frame,  # last_frame
                    self.video_manager.background,  # background
                    self.video_manager.background_angle_diff,  # angle_diff
                    task_name="add_sample",
                    wait_result=False  # 不等待结果
                )
            )

            # 异步执行特征保存
            asyncio.create_task(
                self._run_with_error_handling(
                    self.save_game_timeseries_features,
                    game,
                    task_name="save_game_timeseries_features",
                    wait_result=False  # 不等待结果
                )
            )

            # 检查是否有帧数据可用于特征提取
            if game.frames is not None and len(game.frames) > 0:
                # 异步执行帧提取和特征保存
                asyncio.create_task(
                    self._save_features_with_frame_extraction(game)
                )
            elif hasattr(game, '_frames_future') and game._frames_future is not None:
                # 如果帧还在提取中，等待提取完成后再处理
                asyncio.create_task(
                    self._wait_for_frames_and_save_features(game)
                )

        else:
            self.logger.warning(f"未找到游戏：{data['body']['seqNo']}")
            # 尝试恢复游戏状态
            recovered_game = self.game_manager.recover_game_state(data['body']['seqNo'])
            if recovered_game:
                self.logger.info(f"已恢复游戏状态: {recovered_game.to_string()}")
                # 重新处理结果消息
                recovered_game.update_status(GameStatus.HANDLE_RESULT)
                recovered_game.result = data['body']['result'][0]
                self.logger.info(f"处理恢复游戏结果: {recovered_game.to_string()}")
            else:
                # 记录待处理消息，等待游戏创建
                self.pending_recovery[data['body']['seqNo']] = {
                    'type': 'result',
                    'data': data,
                    'timestamp': time_module.time()
                }
                
    async def _wait_for_frames_and_save_features(self, game):
        """
        等待帧提取完成后再保存特征
        """
        try:
            if hasattr(game, '_frames_future') and game._frames_future is not None:
                # 在线程中等待future完成
                loop = asyncio.get_running_loop()
                frames = await loop.run_in_executor(None, self._wait_for_future, game._frames_future)
                if frames is not None and len(frames) > 0:
                    game.frames = frames
                    # 异步执行帧提取和特征保存
                    asyncio.create_task(
                        self._save_features_with_frame_extraction(game)
                    )
        except Exception as e:
            self.logger.error(f"等待帧完成时出错: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def _save_features_with_frame_extraction(self, game):
        try:
            _end_frames = await self.video_manager.extract_frames_for_period(
                game.seq_no, count=6, step=0.5)
            # 获取背景图和角度差
            backgrounds = self.video_manager.background_history[-5:] if len(
                self.video_manager.background_history) >= 5 else self.video_manager.background_history.copy()
            if self.video_manager.background is not None:
                backgrounds.append(self.video_manager.background)
                
            background_angle_diffs = []
            for bg in backgrounds:
                # 这里应该计算每个背景的角度差，简化处理
                background_angle_diffs.append(self.video_manager.background_angle_diff)
                
            self.feature_extractor.save_all_features(
                game, 
                game.frames, 
                _end_frames, 
                backgrounds, 
                background_angle_diffs,
                self.logger
            )
        except Exception as e:
            self.logger.error(f"Error in _save_features_with_frame_extraction: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def handle_bet_statement(self, data):
        # {"cmd":2,"cmdName":"GAME_CMD","action":35,"actionName":"BETTING_STAT_ACT","size":108,"ext":0,"body":{"type":6,"gameTypeName":"SPEED_SICBO","tableID":"B21","count":5,"playerTotal":0,"chipsTotal":0,"chips":[{"num":7,"type":1610612737,"betTypeName":"BIG","amt":3177.255997657776},{"num":2,"type":1610612738,"betTypeName":"SMALL","amt":16.170059949159622},{"num":0,"type":1610612740,"betTypeName":"ODD","amt":0},{"num":2,"type":1610612744,"betTypeName":"EVEN","amt":180},{"num":4,"type":1610612752,"betTypeName":"DICE","amt":417.72099912166595}]}}
        if data['body']['tableID'] != self.table_id:
            return None
            
        game = self.game_manager.get_latest_game()
        if game is not None:
            chips = data['body']['chips']
            game.chips = chips
        else:
            self.logger.warning("收到下注统计消息但未找到游戏")

    async def _run_with_error_handling(self, func, *args, task_name="Task", wait_result=True):
        """带错误处理的线程池任务执行"""
        loop = asyncio.get_running_loop()
        try:
            if wait_result:
                await loop.run_in_executor(self.thread_pool, func, *args)
            else:
                # 不等待结果，直接在后台执行
                asyncio.create_task(self._run_in_executor_no_wait(loop, func, *args, task_name=task_name))
        except Exception as e:
            self.logger.error(f"Error in {task_name}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def _run_in_executor_no_wait(self, loop, func, *args, task_name="Task"):
        """在后台执行线程池任务，不等待结果"""
        try:
            await loop.run_in_executor(self.thread_pool, func, *args)
            self.logger.debug(f"Completed background task: {task_name}")
        except Exception as e:
            self.logger.error(f"Error in background task {task_name}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
    async def save_game_timeseries_features(self, game):
        """保存游戏的时间序列特征（异步执行）"""
        loop = asyncio.get_running_loop()
        # 将特征提取和保存操作放入线程池，避免阻塞事件循环
        await loop.run_in_executor(
            self.thread_pool,
            self._sync_save_game_timeseries_features,
            game
        )

    def _sync_save_game_timeseries_features(self, game):
        """同步版本的特征保存，供线程池调用"""
        # 这里使用最新的背景图进行特征提取
        input_features = self.feature_extractor.to_input_feature(game, self.video_manager.processor)
        if input_features is not None:
            self.timeseries_storage.save_features(input_features, game)
            
    async def statement(self):
        all_count = 0
        max_win = 0
        min_win = 0
        total_win = 0
        games = self.game_manager.get_games()
        for game in games:
            if game.recommend is not None:
                all_count += 1
                win = game.win()
                total_win += win
                if max_win < total_win:
                    max_win = total_win
                if min_win > total_win:
                    min_win = total_win
        win_rate = 0
        if all_count != 0:
            win_rate = round(total_win / all_count * 100, 2)
        return f"盈利率：{win_rate}% 下注总量:{all_count} 最大盈利:{max_win:.2f} 最小盈利:{min_win:.2f} 当前盈利:{total_win:.2f}"