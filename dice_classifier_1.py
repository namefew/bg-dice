import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import logging
from torch.optim.lr_scheduler import StepLR

from video_processor import DiceVideoProcessor

dice_classifier = None


def get_cnn_instance():
    global dice_classifier
    if dice_classifier is None:
        dice_classifier = DiceClassifier()
    return dice_classifier

# 配置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 自定义数据集类
class DiceDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        # 直接加载数据到内存
        self.data_arrays = [np.load(os.path.join(folder_path, f)) for f in self.files]
        # 保持原有统计量计算逻辑...
        self.mean, self.std = self._calculate_stats(folder_path)
        # 启用内存映射
        # self.mmaps = [np.load(f'{folder_path}/{f}', mmap_mode='r') for f in self.files]

    def __len__(self):
        return len(self.files)

    # 修正后的统计量计算逻辑
    def _calculate_stats(self, folder_path):

        train_files = self.files[:int(0.7 * len(self.files))]  # 仅使用训练集部分
        sampled_files = np.random.choice(train_files,
                                         size=min(2000, len(train_files)), replace=False)
        all_features = []
        for f in sampled_files[:1000]:
            array = np.load(os.path.join(folder_path, f))
            all_features.append(array[:-1])
        return np.mean(all_features, axis=0), np.std(all_features, axis=0) + 1e-6

    def __getitem__(self, idx):
        array = self.data_arrays[idx]  # 创建可写副本
        label = array[-1]
        feature_vector = array[:-1].copy()  # 显式创建副本
        #feature_vector += np.random.normal(0, 0.1, size=feature_vector.shape)
        # feature_vector = array[:532]
        # label = int(array[-1])
        label = int(self.files[idx].split('_')[0])-1  # 标签从0开始
        # 添加数据增强
        if np.random.rand() > 0.5:
            feature_vector += np.random.normal(0, 0.01, size=feature_vector.shape)
        feature_vector = (feature_vector - self.mean) / self.std
        return torch.tensor(feature_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        # x = array[4] + array[0]  # 根据实际特征位置调整索引
        # y = array[5] + array[1]
        # return torch.tensor([x, y,array[2],array[3],array[6]], dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def plot_all_label_heatmaps(self):
        labels = [0, 1, 2]
        fig, axs = plt.subplots(1, len(labels), figsize=(30, 8))

        for i, label in enumerate(labels):
            x_coords = []
            y_coords = []

            for j in range(len(self)):
                sample = self[j]
                if sample['label'].item() == label:
                    x, y = sample['coordinates']
                    x_coords.append(x)
                    y_coords.append(y)

            # 绘制热力图
            hb = axs[i].hexbin(x_coords, y_coords,
                               gridsize=56,
                               cmap='Oranges',  # 更改为Blues颜色映射
                               mincnt=1,
                               extent=[0, 224, 0, 224])  # 设置x和y的范围

            # 使用fig.colorbar来添加颜色条
            cbar = fig.colorbar(hb, ax=axs[i], label='Point Density')

            axs[i].set_title(f'Coordinate Distribution for Label={label}')
            axs[i].set_xlabel('X Coordinate')
            axs[i].set_ylabel('Y Coordinate')

            # 反转y轴方向
            axs[i].invert_yaxis()

        plt.tight_layout()
        plt.show()

    def get_heatmap_data(self):
        labels = [0, 1, 2]
        heatmap_data = []

        for label in labels:
            x_coords = []
            y_coords = []

            for j in range(len(self)):
                sample = self[j]
                if sample['label'].item() == label:
                    x, y = sample['coordinates']
                    x_coords.append(x)
                    y_coords.append(y)

            # 计算热力图数据
            hb = plt.hexbin(x_coords, y_coords,
                            gridsize=112,
                            mincnt=1,
                            extent=[0, 224, 0, 224])

            # 获取每个单元格的密度值
            counts = hb.get_array()
            offsets = hb.get_offsets()

            # 将密度值和坐标存储到列表中
            data = []
            for i, count in enumerate(counts):
                x_center, y_center = offsets[i]
                data.append([x_center, y_center, count])

            heatmap_data.append(data)

        return heatmap_data

    def plot_all_label_heatmaps_to_table(self):
        heatmap_data = self.get_heatmap_data()
        labels = [0, 1, 2]

        for i, label in enumerate(labels):
            df = pd.DataFrame(heatmap_data[i], columns=['X Coordinate', 'Y Coordinate', 'Point Density'])
            print(f"Coordinate Distribution for Label={label}")
            print(df.head())  # 打印前几行数据

            # 保存为CSV文件
            df.to_csv(f'label_{label}_heatmap_data.csv', index=False)


    def __getitem__0(self, idx):
        array = self.data_arrays[idx]
        # 假设特征向量结构为：[x, y, 其他特征..., 当前点数]
        x = array[4]+array[0]  # 根据实际特征位置调整索引
        y = array[5]+array[1]
        current_dot = int(self.files[idx].split('_')[1][0])  # 倒数第二个位置是当前点数
        next_dot = int(self.files[idx].split('_')[0])  # 文件名中的目标点数
        label = 0
        if current_dot==next_dot:
            label = 1
        elif current_dot+next_dot == 7:
            label = 2
        return {
            'label': torch.tensor(label, dtype=torch.long),
            'coordinates': (x, y),
            'shap':(array[2],array[3])
        }

class TransitionModel(nn.Module):
    def __init__(self, input_dim=272):  # 修改输入维度为4
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),  # 添加批量归一化
            nn.GELU(),
            nn.Dropout(0.5),  # 增强正则化

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128, 6)
        )
        self._init_weights()
        # 初始化权重

    def _init_weights(self):
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # 将nonlinearity参数改为relu（PyTorch官方支持）
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    # 或者使用更普适的初始化方式
                    # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('gelu'))
                    nn.init.constant_(m.bias, 0.1)
    def forward(self, x):
        return self.model(x)



class DiceClassifier:
    def __init__(self, input_size=272,  # 修改输入维度为4
                 hidden_sizes=[256,128,64],
                 num_classes=6,
                 batch_size=32,
                 num_epochs=100,
                 learning_rate=0.0001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.mean = None
        self.std = None
        # 模型、损失函数和优化器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransitionModel().to(self.device)  # 修改模型输入维度
        # 使用更好的优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate,
                                     weight_decay=1e-4)  # 调整权重衰减
        # 改进学习率调度
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=50)
        # 添加类别权重处理
        self.criterion = nn.CrossEntropyLoss()
        weight_path = self.weight_path = 'dice_transition1_model.pth'
        if os.path.exists(weight_path):
            self.load_model(weight_path)
            logging.info(f"Loaded model weights from {weight_path}")
        else:
            logging.info(f"Model weights file {weight_path} not found. Starting with random weights.")

    def train(self,folder_path ):
        self.raw_dataset = DiceDataset(folder_path)  # 保存完整数据集
        dataset = self.raw_dataset  # 使用完整数据集

        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        # 在初始化中添加

        self.early_stop_patience = 10
        self.best_epoch = 0
        best_val_acc = 0.0
        self.model.train()
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else torch.amp.GradScaler('cpu')
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 新增梯度裁剪
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

            # ===== 验证阶段 =====
            self.model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # 计算验证指标
            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_acc = val_correct / val_total
            # 记录日志（添加验证指标）
            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                         f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | '
                         f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

            # 根据验证指标保存最佳模型
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                torch.save(self.model.state_dict(), self.weight_path)
                logging.info(f'New best model saved with val acc: {best_val_acc:.4f}')

            # scheduler.step()  # 更新学习率 #StepLR是按固定周期调整，而ReduceLROnPlateau依赖于某个指标的变化

            self.scheduler.step(val_epoch_acc)
        self.analyze_features(self.raw_dataset)
        self.test(test_loader)

    # 在DiceClassifier中添加
    def analyze_features(self, dataset):
        # 创建包含多个样本的批量数据
        sample_indices = list(range(min(32, len(dataset))))  # 使用前32个样本
        batch = [dataset[i] for i in sample_indices]
        inputs = torch.stack([x for x, _ in batch]).to(self.device)

        # 保存原始模式并切换为评估模式
        original_mode = self.model.training
        self.model.eval()

        with torch.no_grad():
            activations = []

            def get_activation(name):
                def hook(model, input, output):
                    activations.append(output.detach().cpu())

                return hook

            # 注册hook以获取每一层的输出
            hooks = []
            for name, layer in self.model.named_modules():
                if isinstance(layer, nn.Linear):  # 只收集线性层的输出
                    hook = layer.register_forward_hook(get_activation(name))
                    hooks.append(hook)

            _ = self.model(inputs)  # 输入批量数据

            # 移除hook
            for hook in hooks:
                hook.remove()

            # 绘制每个样本的激活分布
            plt.figure(figsize=(12, 6))
            for i, act in enumerate(activations):
                plt.subplot(1, len(activations), i + 1)
                plt.hist(act.flatten(), bins=50, alpha=0.5)
                plt.title(f'Layer {i + 1}')
            plt.show()

        # 恢复原始训练模式
        self.model.train(original_mode)

    def test(self,test_loader):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 确保数据在同一个设备上
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_accuracy = correct / total
            logging.info(f'Test Accuracy: {test_accuracy:.4f}')
            print(f'Test Accuracy: {test_accuracy:.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    def predict(self, frame:np.ndarray, background):

        video_processor = DiceVideoProcessor(background)
        features = video_processor.detect_dice_feature(frame)
        if features is None:
            return [], []
        features = (features - self.mean) / self.std
        # 添加batch维度
        x = features[4] + features[0]  # 根据实际特征位置调整索引
        y = features[5] + features[1]
        input_tensor = torch.tensor([x, y, features[2], features[3], features[6]], dtype=torch.float32).unsqueeze(0).to(self.device)

        # input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # 移除batch维度
            probs = torch.softmax(outputs.squeeze(0), dim=-1)
            max_prob, pred = torch.max(probs, dim=-1)
            return pred.item()+1, max_prob.item()
    def predict_image_top(self, frame: np.ndarray, background=None, n=6):
        if background is None:
            return [], []
        video_processor = DiceVideoProcessor(background)
        features = video_processor.detect_dice_feature(frame)
        if features is None:
            return [], []
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)  # 移除了隐藏状态的接收
            if outputs.dim() == 2:  # 现在输出是(batch=1, classes)
                outputs = outputs.squeeze(0)

            softmax = nn.Softmax(dim=0)
            probabilities = softmax(outputs).cpu().numpy()

            # 获取前N个结果
            topN_indices = np.argpartition(probabilities, -n)[-n:]
            topN_class = topN_indices[np.argsort(probabilities[topN_indices])][::-1]
            topN_prob = probabilities[topN_class]

        return topN_class+1, topN_prob


if __name__ == "__main__":
    # dataset = DiceDataset(folder_path='train/features')
    # dataset.plot_all_label_heatmaps()
    # dataset.plot_all_label_heatmaps_to_table()

    # dataset.plot_dot_distribution()
    classifier = DiceClassifier()
    classifier.train(folder_path='train/features')

