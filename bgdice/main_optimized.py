import asyncio
import config
from bgdice.dice_server_optimized import DiceServerOptimized


# 启动WebSocket服务
if __name__ == "__main__":
    config.start_file_watcher()
    server = DiceServerOptimized(port=config.get_instance().get('server_port', 8765))
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Server stopped by user")