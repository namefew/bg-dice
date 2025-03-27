from collections import Counter

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

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dice_classifier = None


def get_cnn_instance():
    global dice_classifier
    if dice_classifier is None:
        dice_classifier = CNN()
    return dice_classifier


# 定义数据集类
class DiceDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_augmentations=1):
        self.root_dir = root_dir
        self.transform = transform
        self.num_augmentations = num_augmentations
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images) * self.num_augmentations

    def __getitem__(self, idx):
        original_idx = idx // self.num_augmentations
        img_path = os.path.join(self.root_dir, self.images[original_idx])
        image = Image.open(img_path).convert('RGB')  # 使用 PIL.Image 打开图片
        label = int(self.images[original_idx].split('_')[0]) - 1  # 标签从0开始
        label = 0 if label<4 else 1
        if self.transform:
            image = self.transform(image)

        return image, label


# 定义模型
class DiceModel(nn.Module):
    def __init__(self, num_classes=6):
        super(DiceModel, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features

        # 修改最后一层
        # self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),  # 添加隐藏层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout(0.5),  # 添加Dropout防止过拟合
            nn.Linear(512, num_classes)  # 最终输出层
        )

    def forward(self, x):
        return self.resnet(x)

    def extract_features(self, x):
        # 提取第 n-1 层的特征向量
        return self.resnet.layer4(self.resnet.layer3(self.resnet.layer2(self.resnet.layer1(self.resnet.conv1(x)))))


class CNN():
    def __init__(self):

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 80, img.width, img.height))),  # 新增：裁剪顶部80像素
            transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
            transforms.Lambda(self._normalize_lighting),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiceModel(num_classes=2).to(self.device)
        # 初始化损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        resnet_params = []
        fc_params = []
        for name, param in self.model.resnet.named_parameters():
            if 'fc' in name:
                fc_params.append(param)
            else:
                resnet_params.append(param)

        # self.optimizer = optim.Adam([
        #     {'params': [p for p in resnet_params if p.requires_grad], 'lr': 0.0001},
        #     {'params': fc_params, 'lr': 0.001}
        # ])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=2e-4,  # 适当提高初始学习率
            weight_decay=1e-4  # 增加权重衰减
        )

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode='max',
        #     factor=0.5,
        #     patience=3,
        #     verbose=True
        # )
        # 改用CosineAnnealing调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=20,  # 半周期长度
            eta_min=1e-6
        )
        # 检查权重文件是否存在
        weight_path = self.weight_path = 'bg_dice_predict_big_small.pth'
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device), strict=False )
            logging.info(f"Loaded model weights from {weight_path}")
            # # 冻结前 6 层  #保留 Layer3 和 Layer4 可训练
            # for i, child in enumerate(self.model.resnet.children()):
            #     if i < 6:
            #         for param in child.parameters():
            #             param.requires_grad = False
            # 冻结所有层除了fc
            # for name, param in self.model.resnet.named_parameters():
            #     if 'fc' not in name:  # 关键修改点
            #         param.requires_grad = False
            #     else:
            #         param.requires_grad = True  # 显式启用fc层

            for name, param in self.model.resnet.named_parameters():
                if 'layer3' in name or 'layer4' in name:
                    param.requires_grad = True
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

    # 训练模型
    def _train_model(self, model, criterion, optimizer, scheduler, num_epochs=50, folder_path='train/new-images'):
        # 加载数据集
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 80, img.width, img.height))),  # 新增：裁剪顶部80像素
            transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
            transforms.Lambda(self._normalize_lighting),  # 光照归一化
            # transforms.RandomRotation(degrees=10),  # 限制旋转角度
            # transforms.RandomAffine(degrees=(-10, 10), translate=(0, 0.2), scale=(0.8, 1.1)),  # 只允许向下平移
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增加光照变换
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = DiceDataset(root_dir=folder_path, transform=train_transform, num_augmentations=1)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        model.train()
        # 添加最佳验证指标跟踪
        best_val_acc = 0.0
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else torch.amp.GradScaler('cpu')

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                    outputs = model(images)
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
            # ===== 验证阶段 =====
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # 计算验证指标
            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_acc = val_correct / val_total

            # 记录日志（添加验证指标）
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], '
                         f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | '
                         f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

            # 根据验证指标保存最佳模型
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                torch.save(model.state_dict(), self.weight_path)
                logging.info(f'New best model saved with val acc: {best_val_acc:.4f}')

            scheduler.step(val_epoch_acc)  # 更新学习率
        torch.save(model.state_dict(), self.weight_path.replace('.pth','_last.pth'))

    # 识别图片
    def predict_image_path(self, image_path: str):
        """
        参数:
            image_path: 输入图像文件路径。
        返回:
            predicted_class: 预测的类别。
            confidence: 预测的置信度。
        """
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(outputs).squeeze().cpu().numpy()
            predicted_class = predicted.item()
            confidence = probabilities[predicted_class]
        return predicted_class+1, confidence

    def predict_image(self, image: np.ndarray):
        """
        预测给定图像的类别。
        参数:
            image: 输入图像的 NumPy 数组。
        返回:
            predicted_class: 预测的骰子点数。
            confidence: 预测的置信度。
        """
        self.model.eval()
        # 将 NumPy 数组转换为 PIL 图像
        image_pil = Image.fromarray(image.astype(np.uint8))
        image_pil = image_pil.convert('RGB')
        # 应用数据变换
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(outputs).squeeze().cpu().numpy()
            predicted_class = predicted.item()
            confidence = probabilities[predicted_class]

        return predicted_class, confidence


    def predict_image_top(self, frame: np.ndarray,  background=None):
        cls,cof = self.predict_image(frame)
        return  [1,2,3,4,5,6] if cls==0 else [6,5,4,3,2,1] ,[cof,0,0,0,0,1-cof]
    def extract_features_from_image(self, image: np.ndarray):
        """
        从 NumPy 数组提取特征向量。
        参数:
            image: 输入图像的 NumPy 数组。
        返回:
            features: 特征向量。
        """
        self.model.eval()
        # 将 NumPy 数组转换为 PIL 图像
        image_pil = Image.fromarray(image.astype(np.uint8))
        image_pil = image_pil.convert('RGB')
        # 应用数据变换
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.extract_features(image_tensor)
        return features.squeeze().cpu().numpy()

    def _visualize_transformed_images(self, dataset, num_samples=4):
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 20))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in range(num_samples):
            original_idx = i // dataset.num_augmentations
            img_path = os.path.join(dataset.root_dir, dataset.images[original_idx])
            original_image = Image.open(img_path).convert('RGB')
            transformed_image, _ = dataset[i * dataset.num_augmentations]

            # 反归一化
            transformed_image = transformed_image * std + mean
            transformed_image = transforms.ToPILImage()(transformed_image)

            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(transformed_image)
            axes[i, 1].set_title('Transformed Image')
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def train(self, num_epochs=100,folder_path='train/new-images'):
        # 可视化增强后的图像
        # self._visualize_transformed_images(self.train_dataset, num_samples=5)
        # 添加数据分布可视化
        label_counts = Counter([int(f.split('_')[0]) for f in os.listdir(folder_path)])
        plt.bar(label_counts.keys(), label_counts.values())
        plt.title('Class Distribution')
        plt.show()
        # 继续训练模型
        self._train_model(self.model, self.criterion, self.optimizer, self.scheduler, num_epochs=num_epochs,folder_path=folder_path)
        # 保存模型

    def test(self):
        # 识别 images 文件夹中 m_ 开头的图片
        image_dir = 'train/new_val-0'
        total = 0
        correct = 0
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg'):
                total += 1
                image_path = os.path.join(image_dir, filename)
                predicted_class, confidence = self.predict_image_path(image_path)
                logging.info(f'File: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence:.4f}')
                # if confidence > 0.90:
                new_filename = f'{predicted_class + 1}_{filename[2:]}'
                if filename == new_filename:
                    correct += 1
                    continue
                new_image_path = os.path.join(image_dir, new_filename)
                # os.rename(image_path, new_image_path)
                logging.info(f'Renamed to: {new_filename}')
        logging.info(f'Accuracy: {correct}/{total} = {correct / total:.4f}')


# 程序入口
if __name__ == "__main__":
    cnn = get_cnn_instance()
    cnn.train(num_epochs=100,folder_path='train/new-images')  # 启用训练
    # cnn.test()
    # predicted_class, confidence = cnn.predict_image_path('output/dice_roi1742046702.3200257.jpg')
    # print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.4f}')