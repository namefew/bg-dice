import asyncio
import websockets
import logging

# 配置日志记录到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('websocket_server.log', mode='a'),
        logging.StreamHandler()  # 添加控制台输出
    ]
)


class WebSocketServer:
    def __init__(self, port=8765):
        self.port = port
        self.clients = set()
        self.lock = asyncio.Lock()
        self.sent_clients = set()  # 用于存储已发送过消息的客户端
        self.messages = {}

    async def handler(self, websocket):
        async with self.lock:
            self.clients.add(websocket)
            logging.info(f"New client connected: {websocket.remote_address}, total {len(self.clients)}")
        try:
            async for message in websocket:
                logging.info(f"Received message: {message}  from:[{websocket.remote_address}]:")
                if not message.startswith('{') and not message.endswith('}'):
                    if websocket in self.clients:
                        self.clients.remove(websocket)
                        self.sent_clients.add(websocket)
                    await self.broadcast(message, exclude=[websocket])
                else:
                    await self.handleJsonMessage(message, websocket)
        except websockets.exceptions.ConnectionClosedError:
            pass
        except Exception as e:
            logging.warning(f"Error handling message from [{websocket.remote_address}]: {e}")
        finally:
            async with self.lock:
                if websocket in self.sent_clients:
                    self.sent_clients.remove(websocket)
                if websocket in self.clients:
                    self.clients.remove(websocket)
                logging.info(f"Client disconnected: {websocket.remote_address}")

    async def broadcast(self, message, exclude=None):
        async with self.lock:
            if not self.validate(message):
                logging.info(f"Duplicate message detected: {message}. Skipping broadcast.")
                return
            logging.info(f"Broadcasting message to {len(self.clients)} clients: {message}")
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
                logging.info(f"Received UDP message from {addr}: {message}")
                # 创建任务异步处理消息
                asyncio.create_task(self.server.broadcast(message))

        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self),
            local_addr=('0.0.0.0', 5005)
        )
        logging.info(f"UDP listener started on port 5005")

    # 修改start方法
    async def start(self):
        async with websockets.serve(self.handler, "0.0.0.0", self.port):
            logging.info(f"WebSocket server started on port {self.port}")
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

    async def handleJsonMessage(self, message, websocket):
        pass


# 启动WebSocket服务
if __name__ == "__main__":
    server = WebSocketServer(port=8765)
    asyncio.run(server.start())
