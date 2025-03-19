import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image  # 明确导入 PIL.Image
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from torchvision.models import resnet18, ResNet18_Weights

MODEL_RESNET_PTH = 'bg_model_resnet18_rnn.pth'

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DiceModel(nn.Module):
    def __init__(self, num_classes=7, input_size=512, hidden_size=256, num_layers=2):
        super(DiceModel, self).__init__()

        # 使用预训练的 ResNet-18 提取图像特征
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, input_size)  # 修改最后一层为指定大小

        # 定义 RNN 层
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # 定义全连接层输出分类结果
        self.fc = nn.Linear(hidden_size, num_classes)

        # 添加激活函数和 dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)

        # 提取图像特征
        features = self.resnet(x)
        features = self.relu(features)
        features = self.dropout(features)

        # 调整形状以适应 RNN 输入
        features = features.view(batch_size, seq_length, -1)

        # 通过 RNN 层
        out, _ = self.rnn(features)

        # 获取最后一个时间步的输出
        out = out[:, -1, :]

        # 通过全连接层得到最终输出
        out = self.fc(out)

        return out


class DiceDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_length=5):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.cached_images = {}  # 缓存图像

    def __len__(self):
        return len(self.images) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = (idx + 1) * self.seq_length
        image_paths = [os.path.join(self.root_dir, self.images[i]) for i in range(start_idx, end_idx)]

        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)

        # 假设标签从文件名中提取（这里简化为第一个帧的标签）
        label = int(self.images[start_idx].split('_')[0])

        return images, label


class CNN():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
            transforms.Lambda(self._normalize_lighting),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增加光照变换
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiceModel(num_classes=7).to(self.device)

        # 初始化损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3)

        weight_path = MODEL_RESNET_PTH
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            logging.info(f"Loaded model weights from {weight_path}")
        else:
            logging.info(f"Model weights file {weight_path} not found. Starting with random weights.")

    def _normalize_lighting(self, image):
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        image_np[:, :, 0] = cv2.equalizeHist(image_np[:, :, 0])
        image_np = cv2.cvtColor(image_np, cv2.COLOR_LAB2RGB)
        return Image.fromarray(image_np)

    # 自定义噪声添加函数
    def _add_gaussian_noise(self, image, mean=0, std=0.1):
        np_image = np.array(image) / 255.0
        noise = np.random.normal(mean, std, np_image.shape)
        noisy_image = np.clip(np_image + noise, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8))

    def _train_model(self, model, criterion, optimizer, scheduler, num_epochs=50):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
            transforms.Lambda(self._normalize_lighting),  # 光照归一化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = DiceDataset(root_dir='train/new_images', transform=train_transform, seq_length=5)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
            transforms.Lambda(self._normalize_lighting),  # 光照归一化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = DiceDataset(root_dir='train/val', transform=val_transform, seq_length=5)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

        model.train()
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else torch.amp.GradScaler('cpu')

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

            # 验证集评估
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_sequences, val_labels in val_loader:
                    val_sequences, val_labels = val_sequences.to(self.device), val_labels.to(self.device)
                    val_outputs = model(val_sequences)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()

            val_acc = val_correct / val_total
            logging.info(f'Validation Accuracy: {val_acc:.4f}')

            # 更新学习率
            scheduler.step(val_acc)

            # 保存模型
            torch.save(model.state_dict(), MODEL_RESNET_PTH)

# 程序入口
if __name__ == "__main__":
    cnn = CNN()
    cnn._train_model(cnn.model, cnn.criterion, cnn.optimizer, cnn.scheduler, num_epochs=50)
    # cnn.test()