import logging

import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from bg_dice_resnet18 import CNN
from logger import Logger
from online_video_processor import DiceOnlineVideoProcessor
import pyautogui  # 导入 pyautogui 库

class DiceApp:
    def __init__(self, root):
        self.logger = Logger(log_file="app.log")

        self.root = root
        self.root.title("Dice Video Processor")
        self.cnn = CNN()
        self.roi = [514, 134, 224, 224]
        self.save_frame_count = 0
        self.last_second = None
        self.processor = DiceOnlineVideoProcessor(roi=self.roi,logger=self.logger)
        self.processor.add_next_frame_callback(self.process_frame)
        self.running = False
        self.url_var = tk.StringVar()
        self.create_widgets()
        self.next = -1
        self.win = 0
        self.count = 0

    def create_widgets(self):
        # URL输入框
        self.url_label = ttk.Label(self.root, text="输入视频地址:")
        self.url_label.grid(row=0, column=0, padx=10, pady=10)
        self.url_entry = ttk.Entry(self.root, textvariable=self.url_var, width=50)
        self.url_entry.grid(row=0, column=1, padx=10, pady=10)

        # 开始按钮
        self.start_button = ttk.Button(self.root, text="Start", command=self.start_processing)
        self.start_button.grid(row=0, column=2, padx=10, pady=10)

        # # 图像显示区域
        self.image_label = ttk.Label(self.root)
        self.image_label.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

        # 预测点数标签
        self.dot_label = ttk.Label(self.root, text="预测: ")
        self.dot_label.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

    def start_processing(self):
        if self.running:
            self.logger.info("Stopping processing...")
            self.running = False
            self.start_button.config(text="Start")
        else:
            self.logger.info("Starting processing...")
            self.running = True
            self.start_button.config(text="Stop")
            self.win = 0
            self.count = 0
            self.next = -1
        url = self.url_var.get()
        if self.running:
            self.processor.start_process(url)
        else:
            self.processor.stop_process()

    def process_frame(self,frame,second,dot,changed):
        next_dot, confidence = self.cnn.predict_image_top(frame)
        self.dot_label.config(text=f"{second}当前：{dot}预测: {next_dot}预测置信度: {confidence}")
        self.show_image(frame)
        if changed:
            if self.last_second is None or second - self.last_second > 25:
                next = int(next_dot[0])
                if self.next>0:
                    if self.next == dot:
                        self.win += 4.75
                    else:
                        self.win -= 1
                    self.logger.info(f"{second} 预测:{self.next} 实际:{dot}  总次数:{self.count} 总盈利:{self.win}")
                self.next = next
                key_to_press = str(next)
                pyautogui.hotkey('ctrl', key_to_press)
                self.logger.info(f"触发热键: ctrl+{key_to_press}")
                self.last_second = second
                self.count += 1


    def show_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img


if __name__ == "__main__":
    root = tk.Tk()
    app = DiceApp(root)
    root.mainloop()
