# train_dnn_torch.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import pickle


class StandardScaler:
    """标准化器，替代sklearn的StandardScaler"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """计算均值和标准差"""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # 避免除零错误
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        """标准化数据"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """拟合并转换数据"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """反向标准化"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return X * self.scale_ + self.mean_


def train_test_split(X, y, test_size=0.2, random_state=None):
    """数据集划分函数，替代sklearn的train_test_split"""
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    # 创建随机索引
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # 分割数据
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


class FeatureSaver:
    """特征保存器类，用于管理特征数据的保存"""

    def __init__(self, feature_dir="training_data", feature_file="dice_features"):
        self.feature_dir = feature_dir
        self.feature_file = feature_file
        self.cache = {
            'inputs': [],
            'outputs': [],
            'labels': []
        }
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """确保特征目录存在"""
        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir)

    def add_to_cache(self, input_features, output_features, labels):
        """将特征添加到缓存中，用于批量保存"""
        try:
            # 确保输入是numpy数组
            input_features = np.array(input_features, dtype=np.float32)
            output_features = np.array(output_features, dtype=np.float32)
            labels = np.array(labels)

            # 添加到缓存
            self.cache['inputs'].append(input_features)
            self.cache['outputs'].append(output_features)
            self.cache['labels'].append(labels)

            return True
        except Exception as e:
            print(f"Failed to add features to cache: {e}")
            return False

    def flush_cache(self):
        """将缓存中的所有特征一次性保存到文件"""
        if not any(self.cache[key] for key in self.cache):
            return True  # 缓存为空，直接返回

        try:
            # 将缓存中的数据堆叠成数组
            inputs_list = self.cache['inputs']
            outputs_list = self.cache['outputs']
            labels_list = self.cache['labels']

            if not inputs_list:
                return True

            # 重塑输入数据以确保形状一致
            reshaped_inputs = [inp.reshape(1, -1) if inp.ndim == 1 else inp for inp in inputs_list]
            reshaped_outputs = [out.reshape(1, -1) if out.ndim == 1 else out for out in outputs_list]
            reshaped_labels = [label.reshape(1, -1) if label.ndim == 1 else label for label in labels_list]

            # 堆叠所有数据
            all_inputs = np.vstack(reshaped_inputs)
            all_outputs = np.vstack(reshaped_outputs)
            all_labels = np.vstack(reshaped_labels)

            # 保存到文件
            success = self._save_features_impl(all_inputs, all_outputs, all_labels)

            # 清空缓存
            self.cache['inputs'].clear()
            self.cache['outputs'].clear()
            self.cache['labels'].clear()

            return success

        except Exception as e:
            print(f"Failed to flush features cache: {e}")
            return False

    def _save_features_impl(self, all_inputs, all_outputs, all_labels):
        """实际保存特征到文件的实现"""
        try:
            import h5py

            feature_file_path = os.path.join(self.feature_dir, f"{self.feature_file}.h5")

            # 如果文件不存在，创建新文件
            if not os.path.exists(feature_file_path):
                with h5py.File(feature_file_path, 'w') as f:
                    f.create_dataset('inputs', data=all_inputs,
                                     maxshape=(None, all_inputs.shape[1]),
                                     chunks=True, dtype=np.float32)
                    f.create_dataset('outputs', data=all_outputs,
                                     maxshape=(None, all_outputs.shape[1]),
                                     chunks=True, dtype=np.float32)
                    f.create_dataset('labels', data=all_labels,
                                     maxshape=(None, all_labels.shape[1]),
                                     chunks=True)
            else:
                # 追加到现有文件
                with h5py.File(feature_file_path, 'a') as f:
                    current_inputs_size = f['inputs'].shape[0]
                    current_outputs_size = f['outputs'].shape[0]
                    current_labels_size = f['labels'].shape[0]

                    # 扩展数据集大小
                    f['inputs'].resize((current_inputs_size + all_inputs.shape[0], f['inputs'].shape[1]))
                    f['outputs'].resize((current_outputs_size + all_outputs.shape[0], f['outputs'].shape[1]))
                    f['labels'].resize((current_labels_size + all_labels.shape[0], f['labels'].shape[1]))

                    # 添加新数据
                    f['inputs'][current_inputs_size:] = all_inputs
                    f['outputs'][current_outputs_size:] = all_outputs
                    f['labels'][current_labels_size:] = all_labels

            # print(f"Successfully saved {all_inputs.shape[0]} samples to {feature_file_path}")
            return True
        except ImportError:
            # 如果没有安装h5py，则回退到原来的npz方法
            return self._save_features_npz_impl(all_inputs, all_outputs, all_labels)
        except Exception as e:
            print(f"Failed to save features to HDF5: {e}")
            # 出错时也回退到NPZ方法
            return self._save_features_npz_impl(all_inputs, all_outputs, all_labels)

    def _save_features_npz_impl(self, all_inputs, all_outputs, all_labels):
        """NPZ格式的实际保存实现"""
        try:
            # 将特征追加到文件中
            feature_file_path = os.path.join(self.feature_dir, f"{self.feature_file}.npz")

            if os.path.exists(feature_file_path):
                # 如果文件存在，加载现有数据并追加新数据
                with np.load(feature_file_path, allow_pickle=True) as data:
                    existing_inputs = data['inputs']
                    existing_outputs = data['outputs']
                    existing_labels = data['labels']

                # 合并新旧数据
                combined_inputs = np.vstack([existing_inputs, all_inputs])
                combined_outputs = np.vstack([existing_outputs, all_outputs])
                combined_labels = np.vstack([existing_labels, all_labels])
            else:
                # 如果文件不存在，使用新数据
                combined_inputs = all_inputs
                combined_outputs = all_outputs
                combined_labels = all_labels

            # 保存数据
            np.savez(feature_file_path, inputs=combined_inputs, outputs=combined_outputs, labels=combined_labels)
            # print(f"Successfully saved {all_inputs.shape[0]} samples to {feature_file_path}")
            return True
        except Exception as e:
            print(f"Failed to save features to NPZ: {e}")
            return False

    def save_single_feature(self, input_features, output_features, labels):
        """保存单个特征（为了向后兼容）"""
        # 为了向后兼容，仍然提供单个保存的方法
        success = self.add_to_cache(input_features, output_features, labels)
        if success:
            # 立即刷新以保持向后兼容性
            return self.flush_cache()
        return False


class ModelPaths:
    """模型路径管理类"""

    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self._ensure_directory_exists()

    def _ensure_directory_exists(self):
        """确保模型目录存在"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    @property
    def best_model_path(self):
        return os.path.join(self.models_dir, "best_dice_model.pth")

    @property
    def final_model_path(self):
        return os.path.join(self.models_dir, "final_dice_model.pth")

    @property
    def scaler_x_path(self):
        return os.path.join(self.models_dir, "scaler_X.pkl")

    @property
    def scaler_y_path(self):
        return os.path.join(self.models_dir, "scaler_y.pkl")


class DiceDNN(nn.Module):
    """
    深度神经网络模型用于骰子预测
    """

    def __init__(self, input_dim, output_dim):
        super(DiceDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def load_training_data(data_path="training_data/dice_features"):
    """
    加载训练数据，支持HDF5和NPZ格式
    """
    # 检查HDF5文件
    h5_path = data_path + ".h5"
    npz_path = data_path + ".npz"

    if os.path.exists(h5_path):
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                inputs = f['inputs'][:]
                outputs = f['outputs'][:]
                labels = f['labels'][:]
            print(f"Loaded data from HDF5 file: {inputs.shape[0]} samples")
            return inputs.astype(np.float32), outputs.astype(np.float32)
        except ImportError:
            print("h5py not installed, trying NPZ format...")
        except Exception as e:
            print(f"Failed to load HDF5 file: {e}")

    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            inputs = data['inputs']
            outputs = data['outputs']
            labels = data['labels']
            print(f"Loaded data from NPZ file: {inputs.shape[0]} samples")
            return inputs.astype(np.float32), outputs.astype(np.float32)
        except Exception as e:
            print(f"Failed to load NPZ file: {e}")

    raise FileNotFoundError(f"Training data not found at {h5_path} or {npz_path}")


def create_model(input_dim, output_dim):
    """
    创建PyTorch模型
    """
    model = DiceDNN(input_dim, output_dim)
    return model


def save_scaler(scaler, path):
    """
    保存scaler到文件
    """
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path):
    """
    从文件加载scaler
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_model():
    """
    训练模型主函数
    """
    # 初始化路径管理器
    model_paths = ModelPaths()

    # 加载数据
    print("Loading training data...")
    try:
        inputs, outputs = load_training_data()
        print(f"Loaded {inputs.shape[0]} samples")
        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 数据标准化
    print("Scaling data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(inputs)
    y_scaled = scaler_y.fit_transform(outputs)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 创建模型
    print("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(X_train.shape[1], y_train.shape[1]).to(device)
    print(model)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

    # 训练模型
    print("Training model...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    num_epochs = 100

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        # 更新学习率
        scheduler.step(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_paths.best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 保存最终模型和标准化器
    torch.save(model.state_dict(), model_paths.final_model_path)

    # 保存标准化器
    save_scaler(scaler_X, model_paths.scaler_x_path)
    save_scaler(scaler_y, model_paths.scaler_y_path)

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return model, scaler_X, scaler_y


def evaluate_model():
    """
    评估模型性能
    """
    # 初始化路径管理器
    model_paths = ModelPaths()

    # 加载数据
    try:
        inputs, outputs = load_training_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 加载标准化器
    try:
        scaler_X = load_scaler(model_paths.scaler_x_path)
        scaler_y = load_scaler(model_paths.scaler_y_path)
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return

    # 标准化数据
    X_scaled = scaler_X.transform(inputs)

    # 转换为PyTorch张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # 加载模型
    try:
        model = create_model(inputs.shape[1], outputs.shape[1]).to(device)
        model.load_state_dict(torch.load(model_paths.best_model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 预测
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 计算误差
    mse = np.mean((outputs - y_pred) ** 2, axis=0)
    mae = np.mean(np.abs(outputs - y_pred), axis=0)

    print("Model Evaluation:")
    print(f"MSE per output dimension: {mse}")
    print(f"MAE per output dimension: {mae}")
    print(f"Overall MSE: {np.mean(mse):.6f}")
    print(f"Overall MAE: {np.mean(mae):.6f}")


# 全局特征保存器实例
feature_saver = FeatureSaver()


# 为了向后兼容，保留原有的函数接口
def save_features_batch(input_features, output_features, labels):
    """将特征添加到缓存中，用于批量保存"""
    return feature_saver.add_to_cache(input_features, output_features, labels)


def flush_features_cache():
    """将缓存中的所有特征一次性保存到文件"""
    return feature_saver.flush_cache()


def save_features_to_file(input_features, output_features, labels):
    """将单个特征保存到文件（为了向后兼容）"""
    return feature_saver.save_single_feature(input_features, output_features, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DNN for dice prediction using PyTorch')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    args = parser.parse_args()

    if args.evaluate:
        evaluate_model()
    else:
        train_model()
