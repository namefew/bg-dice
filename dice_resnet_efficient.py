import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

MODEL_HYBRID_PTH = 'bg_model_hybrid.pth'

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
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error opening image {img_path}: {e}")
            raise
        label = int(self.images[original_idx].split('_')[0]) - 1  # 标签从0开始

        if self.transform:
            image = self.transform(image)

        return image, label

# 定义混合模型
class HybridModel(nn.Module):
    def __init__(self, num_classes=6):
        super(HybridModel, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 移除最后的全连接层
        self.resnet.fc = nn.Identity()
        self.efficientnet.classifier[1] = nn.Identity()

        # 获取特征维度
        resnet_features = self.resnet(torch.randn(1, 3, 224, 224)).shape[1]
        efficientnet_features = self.efficientnet(torch.randn(1, 3, 224, 224)).shape[1]

        # 新的全连接层
        self.fc = nn.Sequential(
            nn.Linear(resnet_features + efficientnet_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        resnet_features = self.resnet(x)
        efficientnet_features = self.efficientnet(x)
        combined_features = torch.cat((resnet_features, efficientnet_features), dim=1)
        return self.fc(combined_features)

class CNN():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet 和 EfficientNet 都需要 224x224 的输入
            transforms.Lambda(self._normalize_lighting),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridModel(num_classes=6).to(self.device)
        # 初始化损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # 降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        # 检查权重文件是否存在
        weight_path = MODEL_HYBRID_PTH
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            logging.info(f"Loaded model weights from {weight_path}")
        else:
            logging.info(f"Model weights file {weight_path} not found. Starting with random weights.")

        # 添加验证集数据加载器
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(self._normalize_lighting),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = DiceDataset(root_dir='train/new_val-0', transform=val_transform, num_augmentations=1)
        self.val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
                                   prefetch_factor=2, persistent_workers=True)

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
    def _train_model(self, model, criterion, optimizer, scheduler, num_epochs=100):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(self._normalize_lighting),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 加载数据集
        train_dataset = DiceDataset(root_dir='train/new_images-0', transform=train_transform, num_augmentations=2)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True,
                                  prefetch_factor=2, persistent_workers=True)
        model.train()
        max_acc = 0
        scaler = torch.amp.GradScaler('cuda' if self.device.type == 'cuda' else 'cpu')
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
            # 计算验证损失
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

            scheduler.step(val_loss)  # 更新学习率

            if epoch_acc > max_acc:
                max_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_HYBRID_PTH)
        torch.save(model.state_dict(), f'last——{MODEL_HYBRID_PTH}')

    # 识别图片
    def predict_image_path(self, image_path: str):
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
        return predicted_class, confidence

    def predict_image(self, image: np.ndarray):
        self.model.eval()
        image_pil = Image.fromarray(image.astype(np.uint8))
        image_pil = image_pil.convert('RGB')
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(outputs).squeeze().cpu().numpy()
            predicted_class = predicted.item()
            confidence = probabilities[predicted_class]

        return predicted_class, confidence

    def _visualize_transformed_images(self, dataset, num_samples=4):
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 20))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in range(num_samples):
            original_idx = i // dataset.num_augmentations
            img_path = os.path.join(dataset.root_dir, dataset.images[original_idx])
            original_image = Image.open(img_path).convert('RGB')
            transformed_image, _ = dataset[i * dataset.num_augmentations]

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

    def train(self, epochs=100):
        self._train_model(self.model, self.criterion, self.optimizer, self.scheduler, num_epochs=epochs)
        # torch.save(self.model.state_dict(), f'last——{MODEL_HYBRID_PTH}')

    def test(self):
        image_dir = 'train/new_images-0'
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(image_dir, filename)
                predicted_class, confidence = self.predict_image_path(image_path)
                logging.info(f'File: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence:.4f}')
                new_filename = f'{predicted_class}_{filename[2:]}'
                new_image_path = os.path.join(image_dir, new_filename)
                # os.rename(image_path, new_image_path)
                logging.info(f'Renamed to: {new_filename}')

# 程序入口
if __name__ == "__main__":
    cnn = CNN()
    cnn.train(epochs=100)
    # cnn.test()
    # predicted_class, confidence = cnn.predict_image_path('output/dice_roi1742046702.3200257.jpg')
    # print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.4f}')