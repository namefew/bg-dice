import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import logging
from torch.optim.lr_scheduler import StepLR

from video_processor import DiceVideoProcessor

# 配置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 自定义数据集类
class DiceDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_path = os.path.join(self.folder_path, file)
        array = np.load(file_path)
        feature_vector = array[:-1]
        # label = int(array[-1])
        label = int(self.files[idx].split('_')[0])-1  # 标签从0开始

        return torch.tensor(feature_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 定义多层感知器（MLP）模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))  # 添加Dropout层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # 添加Dropout层
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DiceClassifier:
    def __init__(self,input_size=100884,
        hidden_sizes=[512, 256, 128],
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

        # 数据集和数据加载器


        # 模型、损失函数和优化器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_size, hidden_sizes, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加L2正则化
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)  # 学习率调度器
        weight_path = self.weight_path = 'dice_mlp_model.pth'
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            logging.info(f"Loaded model weights from {weight_path}")
        else:
            logging.info(f"Model weights file {weight_path} not found. Starting with random weights.")

    def train(self,folder_path ):
        dataset = DiceDataset(folder_path)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 确保数据在同一个设备上
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_train_loss = running_loss / len(train_loader)
            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_train_loss:.4f}')
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_train_loss:.4f}')
            val_loss, val_accuracy = self.evaluate(val_loader)
            logging.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            if val_loss < best_val_loss and (epoch+1)%10 == 0:
                best_val_loss = val_loss
                self.save_model(f'dice_mlp_model_best.pth')
            self.scheduler.step()  # 更新学习率
        self.save_model(f'dice_mlp_model_last.pth')
        classifier.test(test_loader)
    def evaluate(self,val_loader=None):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 确保数据在同一个设备上
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            return avg_val_loss, val_accuracy

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
        logging.info(f'Model saved to {path}')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logging.info(f'Model loaded from {path}')

    def predict(self, frame:np.ndarray, background):

        video_processor = DiceVideoProcessor(background)
        features = video_processor.detect_dice_feature(frame)
        if features is None:
            return None
        input =  torch.tensor(features, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.item()

    def predict_image_top(self, frame: np.ndarray, background=None, n=6):
        if background is None:
            return [], []
        video_processor = DiceVideoProcessor(background)
        features = video_processor.detect_dice_feature(frame)
        if features is None:
            return [], []
        input = torch.tensor(features, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input)
            # print(f"Outputs shape: {outputs.shape}")  # 打印 outputs 的形状
            if outputs.dim() == 1:
                softmax = nn.Softmax(dim=0)
            else:
                softmax = nn.Softmax(dim=1)
            probabilities = softmax(outputs).squeeze().cpu().numpy()
            # 获取前3个最大概率及其对应的类别
            topN_prob, topN_class = torch.topk(torch.tensor(probabilities), n)
            topN_prob = topN_prob.numpy()
            topN_class = topN_class.numpy()
        return topN_class, topN_prob


# 使用示例
if __name__ == "__main__":
    classifier = DiceClassifier()
    classifier.train(folder_path='train/features')

