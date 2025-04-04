from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigFileHandler(FileSystemEventHandler):
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def on_modified(self, event):
        if event.src_path.endswith('config.yaml'):
            print("配置文件已修改，重新加载...")
            self.analyzer.reload_config()

def start_file_watcher(analyzer):
    event_handler = ConfigFileHandler(analyzer)
    observer = Observer()
    observer.schedule(event_handler, path=str(Path(__file__).parent), recursive=False)
    observer.start()
    print("正在监听配置文件更改...")
    return observer
