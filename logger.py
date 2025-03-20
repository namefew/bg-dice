import logging
import sys
from datetime import datetime

class Logger:
    def __init__(self, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)

        # 禁用默认的根日志处理器
        if not self.logger.handlers:
            self.logger.propagate = False

        # 创建一个格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 创建一个流处理器，用于输出到控制台
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # 如果提供了日志文件路径，则创建一个文件处理器，用于输出到文件
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


# 示例用法
if __name__ == "__main__":
    logger = Logger(log_file="app.log", level=logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
