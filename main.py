import cv2
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
from PIL import Image, ImageTk

import bg_dice_predict_big_small
import dice_game
from logger import Logger
from online_video_processor import DiceOnlineVideoProcessor

class DiceApp:
    def __init__(self, root):
        self.logger = Logger(log_file="app.log")

        self.root = root
        self.root.title("Dice Video Processor")
        self.cnn = bg_dice_predict_big_small.get_cnn_instance()
        self.roi = [514, 134, 224, 224]
        self.save_frame_count = 0
        self.last_second = None
        self.processor = DiceOnlineVideoProcessor(roi=self.roi, logger=self.logger)
        self.processor.add_next_frame_callback(self.process_frame)
        self.running = False
        self.url_var = tk.StringVar()
        self.type_var = tk.StringVar()  # 添加类型变量
        self.create_widgets()
        self.dice_game = dice_game.DiceGame(logger=self.logger)
    def create_widgets(self):
        # URL输入框
        self.url_label = ttk.Label(self.root, text="输入视频地址:")
        self.url_label.grid(row=0, column=0, padx=10, pady=0)
        self.url_entry = ttk.Entry(self.root, textvariable=self.url_var, width=50)
        self.url_entry.grid(row=0, column=1, padx=10, pady=0)

        # 类型下拉框
        self.type_label = ttk.Label(self.root, text="类型:")
        self.type_label.grid(row=1, column=0, padx=10, pady=0)
        self.type_combobox = ttk.Combobox(self.root, textvariable=self.type_var, values=['1子', '2子', '3子', '大/小', '单/双', '大/小&单/双'])
        self.type_combobox.grid(row=1, column=1, padx=10, pady=0)
        self.type_combobox.current(0)  # 设置默认值为第一个选项

        # 开始按钮
        self.start_button = ttk.Button(self.root, text="Start", command=self.start_processing)
        self.start_button.grid(row=1, column=2, padx=10, pady=10)

        # 图像显示区域
        self.image_label = ttk.Label(self.root)
        self.image_label.grid(row=2, column=0, columnspan=6, padx=10, pady=10)

        # 预测点数标签
        self.dot_label = ttk.Label(self.root, text="预测: ")
        self.dot_label.grid(row=3, column=0, columnspan=6, padx=10, pady=10)

    def start_processing(self):
        if self.running:
            self.logger.info("Stopping processing...")
            self.running = False
            self.start_button.config(text="Start")
        else:
            self.logger.info("Starting processing...")
            self.running = True
            self.start_button.config(text="Stop")
            self.dice_game.reset()
        url = self.url_var.get()
        if self.running:
            self.processor.start_process(url)
        else:
            self.processor.stop_process()

    def process_frame(self, frame, second, current_dot, changed):
        predict_dots, confidences = self.cnn.predict_image_top(frame, background=self.processor.background)
        predict_next_dots = [int(pd) for pd in predict_dots]
        predict_confidences = np.around(confidences, decimals=4)
        # 将数组转换为字符串
        confidence_str = np.array2string(predict_confidences, separator=', ',
                                         formatter={'float_kind': lambda x: f"{x:.4f}"})
        # 去掉数组的方括号
        confidence_str = confidence_str.strip('[]')
        self.dot_label.config(text=f"{second}当前：{current_dot}预测: {predict_next_dots}预测置信度: {confidence_str}")
        self.show_image(frame)
        if changed and len(predict_next_dots) > 0:
            if self.last_second is None or second - self.last_second > 25:
                self.dice_game.check_bets(second,self.type_combobox.get(), current_dot, predict_next_dots,predict_confidences,min_exp=0.95)

    def show_image(self, frame):
        # 使用 OpenCV 缩放图像到 640x640
        frame_resized = cv2.resize(frame, (320, 320))

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img


if __name__ == "__main__":
    root = tk.Tk()
    app = DiceApp(root)
    root.mainloop()
