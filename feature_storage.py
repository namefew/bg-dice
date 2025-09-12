import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os


class TimeSeriesFeatureStorage:
    def __init__(self, storage_path="timeseries_features"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.feature_file = os.path.join(storage_path, "timeseries_features.pkl")
        self.metadata_file = os.path.join(storage_path, "metadata.csv")

    def save_features(self, features: np.ndarray,  game: 'DiceGame'):
        """
        保存时间序列特征和标签
        :param features: 特征向量
        :param label: 标签（游戏结果）
        :param game: DiceGame对象
        """

        label = game.result - 1 if game.result is not None else -1  # 结果1 - 6点， 标签就是0-5点
        # 创建元数据记录
        metadata_record = {
            'timestamp': game.start_time.isoformat(),
            'table_id': game.table_id,
            'seq_no': game.seq_no,
            'round_id': game.round_id,
            'result': game.result,  # 实际结果
            'label': label,  # 用于训练的标签
            'recommend': game.recommend,
            'recommend_confidence': game.recommend_confidence,
            'last_game_result': game.last_game_result,
            'feature_vector_length': len(features) if features is not None else 0,
            # 添加 weekday 和 hour 作为元数据
            'weekday': game.start_time.weekday(),
            'hour': game.start_time.hour
        }

        # 保存特征向量和标签
        if features is not None and game.result is not None:
            self._save_feature_vector(game.seq_no, features, label)

        # 保存元数据
        self._save_metadata(metadata_record)

    def _save_feature_vector(self, seq_no: int, features: np.ndarray, label: int):
        """保存单个特征向量和标签"""
        # 加载现有特征字典
        feature_dict = self._load_feature_dict()
        feature_dict[seq_no] = {
            'features': features,
            'label': label,
            'timestamp': datetime.now().isoformat()
        }

        # 保存特征字典
        with open(self.feature_file, 'wb') as f:
            pickle.dump(feature_dict, f)

    def _save_metadata(self, metadata_record: dict):
        """保存元数据到CSV"""
        df_new = pd.DataFrame([metadata_record])

        if os.path.exists(self.metadata_file):
            df_existing = pd.read_csv(self.metadata_file)
            # 修复 FutureWarning: 明确处理空 DataFrame 的情况
            if not df_existing.empty:
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
        else:
            df_combined = df_new

        df_combined.to_csv(self.metadata_file, index=False)

    def _load_feature_dict(self) -> dict:
        """加载特征字典"""
        if os.path.exists(self.feature_file):
            with open(self.feature_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def load_timeseries_data(self) -> tuple:
        """
        加载时间序列数据用于分析
        返回: (特征矩阵, 标签数组, 元数据DataFrame)
        """
        # 加载特征数据
        feature_dict = self._load_feature_dict()
        metadata_df = pd.read_csv(self.metadata_file) if os.path.exists(self.metadata_file) else pd.DataFrame()

        if not feature_dict or metadata_df.empty:
            return None, None, metadata_df

        # 按序列号排序
        seq_numbers = sorted(feature_dict.keys())
        features_list = [feature_dict[seq]['features'] for seq in seq_numbers]
        labels_list = [feature_dict[seq]['label'] for seq in seq_numbers]

        # 转换为特征矩阵和标签数组
        feature_matrix = np.array(features_list)
        labels_array = np.array(labels_list)

        # 确保元数据按相同顺序排列
        if not metadata_df.empty and 'seq_no' in metadata_df.columns:
            metadata_df = metadata_df.set_index('seq_no').loc[seq_numbers].reset_index()

        return feature_matrix, labels_array, metadata_df

    def get_features_by_time_range(self, start_time: datetime, end_time: datetime):
        """
        根据时间范围获取特征数据
        返回: (特征矩阵, 标签数组, 元数据DataFrame)
        """
        metadata_df = pd.read_csv(self.metadata_file) if os.path.exists(self.metadata_file) else pd.DataFrame()
        if metadata_df.empty:
            return None, None, pd.DataFrame()

        # 转换时间戳列
        metadata_df['timestamp'] = pd.to_datetime(metadata_df['timestamp'])

        # 筛选时间范围
        mask = (metadata_df['timestamp'] >= start_time) & (metadata_df['timestamp'] <= end_time)
        filtered_metadata = metadata_df[mask]

        if filtered_metadata.empty:
            return None, None, filtered_metadata

        # 加载对应的特征向量和标签
        feature_dict = self._load_feature_dict()
        features_list = []
        labels_list = []
        valid_seq_nos = []

        for seq_no in filtered_metadata['seq_no']:
            if seq_no in feature_dict:
                features_list.append(feature_dict[seq_no]['features'])
                labels_list.append(feature_dict[seq_no]['label'])
                valid_seq_nos.append(seq_no)

        feature_matrix = np.array(features_list) if features_list else None
        labels_array = np.array(labels_list) if labels_list else None

        # 确保元数据与特征数据对应
        if valid_seq_nos and not filtered_metadata.empty:
            filtered_metadata = filtered_metadata.set_index('seq_no').loc[valid_seq_nos].reset_index()

        return feature_matrix, labels_array, filtered_metadata
