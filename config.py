import json
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

__config = None
def get_instance():
    global __config
    if __config is None:
        __config = Config()
    return __config

def start_file_watcher():
    event_handler = ConfigFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(Path(__file__).parent), recursive=False)
    observer.start()
    print("正在监听配置文件更改...")
    return observer

class ConfigFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.config = get_instance()

    def on_modified(self, event):
        if event.src_path.endswith('config.json'):
            print("配置文件已修改，重新加载...")
            self.config.reload_config()

class Config:
    def __init__(self, file_path='config.json'):
        self.file_path = file_path
        self._config = self.load_config()

    def load_config(self):
        """加载配置文件"""
        config_path = self.file_path
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get(self, key,default_value=None):
        return self._config.get(key, default_value)

    def reload_config(self):
        """重新加载配置文件"""
        self._config = self.load_config()


