import cv2
import numpy as np
import requests
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from bg_dice_resnet import CNN

class DiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dice Video Processor")
        self.cnn = CNN()
        self.roi = [514, 134, 224, 224]
        self.save_frame_count = 0
        self.last_frame = None

        self.url_var = tk.StringVar()

        self.create_widgets()
        self.stable_count = 0

    def create_widgets(self):
        # URL输入框
        self.url_label = ttk.Label(self.root, text="Enter Video URL:")
        self.url_label.grid(row=0, column=0, padx=10, pady=10)
        self.url_entry = ttk.Entry(self.root, textvariable=self.url_var, width=50)
        self.url_entry.grid(row=0, column=1, padx=10, pady=10)

        # 开始按钮
        self.start_button = ttk.Button(self.root, text="Start", command=self.start_processing)
        self.start_button.grid(row=0, column=2, padx=10, pady=10)

        # 图像显示区域
        self.image_label = ttk.Label(self.root)
        self.image_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

        # 预测点数标签
        self.dot_label = ttk.Label(self.root, text="Dot Count: ")
        self.dot_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

    def start_processing(self):
        print('starting')
        url = self.url_var.get()
        self.cap = cv2.VideoCapture(url)
        self.process_frame()

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        x, y, w, h = self.roi
        frame = frame[y:y + h, x:x + w]
        if self.last_frame is None:
            self.last_frame = frame
        else:
            diff = cv2.absdiff(frame, self.last_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            non_zero_pixels = cv2.countNonZero(thresh)
            if non_zero_pixels > 300:  # 假设100个像素的变化可以忽略
                self.stable_count = 0
            else:
                self.stable_count+=1
                if self.stable_count>100:
                    dot = self.cnn.predict_image(frame)
                    self.dot_label.config(text=f"下一个点数: {dot}")
                    self.show_image(frame)
                    self.stable_count = 0

        self.last_frame = frame
        self.root.after(10, self.process_frame)

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
