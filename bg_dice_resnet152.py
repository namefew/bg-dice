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

from torchvision.models import resnet152, ResNet152_Weights

MODEL_RESNET_PTH = 'bg_model_resnet152.pth'

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义数据集类
class DiceDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_augmentations=1):
        self.root_dir = root_dir
        self.transform = transform
        self.num_augmentations = num_augmentations
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.cached_images = {}  # 缓存图像

    def __len__(self):
        return len(self.images) * self.num_augmentations

    def __getitem__(self, idx):
        original_idx = idx // self.num_augmentations
        img_path = os.path.join(self.root_dir, self.images[original_idx])
        image = Image.open(img_path).convert('RGB')
        label = int(self.images[original_idx].split('_')[0])
        # from_to = self.images[original_idx].split('_')[1]
        # from0 = int(from_to.split('-')[0])
        # to0 = int(from_to.split('-')[1])
        # label = from0 * to0  # 标签从0开始

        if self.transform:
            image = self.transform(image)

        return image, label-1


# 定义模型
class DiceModel(nn.Module):
    def __init__(self, num_classes=7):
        super(DiceModel, self).__init__()
        self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        for param in self.resnet.parameters():
            param.requires_grad = False  # 冻结所有层
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # 仅训练最后一层

    def forward(self, x):
        return self.resnet(x)


class CNN():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
            # transforms.Lambda(self._normalize_lighting),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增加光照变换
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DiceModel(num_classes=6).to(self.device)
        # 初始化损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        # 修改优化器配置（CNN类__init__中）
        self.optimizer = optim.Adam(
            [
                {'params': self.model.resnet.fc.parameters(), 'lr': 1e-4}
            ],
            lr=1e-5  # 其他参数的默认学习率
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3)  # 根据准确率动态调整 检查权重文件是否存在
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

    # 训练模型
    def _train_model(self, model, criterion, optimizer, scheduler, num_epochs=50):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
            transforms.Lambda(self._normalize_lighting),  # 光照归一化
            # transforms.RandomHorizontalFlip(),  # 更简单的增强方法
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增加光照变换
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 加载数据集
        train_dataset = DiceDataset(root_dir='train/new_images-0', transform=train_transform, num_augmentations=1)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)  # 增加 num_workers
        # 加载验证集
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
            # transforms.Lambda(self._normalize_lighting),  # 光照归一化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = DiceDataset(root_dir='train/new_val-0', transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
        max_acc = 0
        model.train()
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

            # 验证集评估
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                    val_outputs = model(val_images)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()

            val_acc = val_correct / val_total
            logging.info(f'Validation Accuracy: {val_acc:.4f}')

            # 更新学习率
            scheduler.step(val_acc)  # 传递验证集准确率

            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(self.model.state_dict(), MODEL_RESNET_PTH)

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

    def predict_image_top(self, frame: np.ndarray, n=6):
        self.model.eval()
        # 将 NumPy 数组转换为 PIL 图像
        image_pil = Image.fromarray(frame.astype(np.uint8))
        image_pil = image_pil.convert('RGB')
        # 应用数据变换
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(outputs).squeeze().cpu().numpy()

            # 获取前3个最大概率及其对应的类别
            topN_prob, topN_class = torch.topk(torch.tensor(probabilities), n)
            topN_prob = topN_prob.numpy()
            topN_class = topN_class.numpy()+1

        return topN_class, topN_prob
    def predict_image(self, image: np.ndarray):
        """
        预测给定图像的类别。
        参数:
            image: 输入图像的 NumPy 数组。
        返回:
            predicted_class: 预测的类别。
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

        return predicted_class+1, confidence

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

    def train(self, epochs=20):
        # 可视化增强后的图像
        # self._visualize_transformed_images(num_samples=5)

        # 继续训练模型
        self._train_model(self.model, self.criterion, self.optimizer, self.scheduler, num_epochs=epochs)
        # 保存模型


    def test(self):
        # 识别 images 文件夹中 m_ 开头的图片
        image_dir = 'train/new_images'
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(image_dir, filename)
                predicted_class, confidence = self.predict_image_path(image_path)
                logging.info(f'File: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence:.4f}')
                # if confidence > 0.90:
                new_filename = f'{predicted_class }_{filename[2:]}'
                new_image_path = os.path.join(image_dir, new_filename)
                os.rename(image_path, new_image_path)
                logging.info(f'Renamed to: {new_filename}')


# 程序入口
if __name__ == "__main__":
    cnn = CNN()
    cnn.train(epochs=100)
    # cnn.test()
    # predicted_class, confidence = cnn.predict_image_path('output/dice_roi1742046702.3200257.jpg')
    # print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.4f}')
