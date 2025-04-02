import logging
from colorama import Fore, Style, init

# 初始化 colorama
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.LIGHTBLACK_EX,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, Fore.RESET)
        original_msg = record.msg
        record.msg = f"{log_color}{original_msg}{Style.RESET_ALL}"
        formatted_msg = super().format(record)
        record.msg = original_msg  # 恢复原始消息，避免重复着色
        return formatted_msg

class LogManager:
    _logger = None

    @staticmethod
    def setup():
        if LogManager._logger is None:
            # 创建一个日志记录器
            logger = logging.getLogger('MAIN')
            logger.setLevel(logging.INFO)

            # 避免重复添加处理器
            if not logger.hasHandlers():
                # 创建一个控制台处理器
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)

                # 创建一个自定义格式化器
                formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

                # 将格式化器添加到处理器
                ch.setFormatter(formatter)

                # 将处理器添加到日志记录器
                logger.addHandler(ch)

                # 创建一个文件处理器
                fh = logging.FileHandler('app.log')
                fh.setLevel(logging.DEBUG)
                fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                fh.setFormatter(fh_formatter)
                logger.addHandler(fh)

            LogManager._logger = logger
        return LogManager._logger
