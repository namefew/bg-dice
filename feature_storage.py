import pickle
import numpy as np
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class TimeSeriesFeatureStorage:
    def __init__(self, storage_dir="features"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.feature_cache = []
        self.cache_limit = 100  # 缓存100个特征后再批量写入


    def save_features(self, features, game=None):
        """
        保存特征数据
        :param features: 特征数据
        :param game: 游戏对象（可选）
        """
        # 创建特征记录
        feature_record = {
            'timestamp': datetime.now().isoformat(),
            'features': features.tolist() if isinstance(features, np.ndarray) else features,
        }

        # 如果有游戏对象，添加游戏相关信息
        if game is not None:
            # 计算标签
            label = game.result - 1 if game.result is not None else -1
            feature_record.update({
                'round_id': game.round_id,
                'seq_no': game.seq_no,
                'table_id': game.table_id,
                'game_status': game.status.name if hasattr(game.status, 'name') else str(game.status),
                'result': game.result,
                'label': label,
                'recommend': game.recommend,
                'recommend_confidence': game.recommend_confidence,
                'last_game_result': game.last_game_result,
                'feature_vector_length': len(features) if features is not None else 0,
                'weekday': game.start_time.weekday(),
                'hour': game.start_time.hour
            })

        # 添加到缓存
        self.feature_cache.append(feature_record)

        # 如果缓存达到限制，写入文件
        if len(self.feature_cache) >= self.cache_limit:
            self.flush_cache()






    def flush_cache(self):
        """
        将缓存中的特征数据写入文件
        """
        if not self.feature_cache:
            return

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.storage_dir, f"features_{timestamp}.json")

        try:
            # 写入文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.feature_cache, f, ensure_ascii=False, indent=2)
            
            print(f"保存了 {len(self.feature_cache)} 条特征数据到 {filename}")
            # 清空缓存
            self.feature_cache.clear()
        except Exception as e:
            print(f"保存特征数据时出错: {e}")

    def save_features_batch(self, input_features, output_features, labels=None):
        """
        批量保存特征数据
        :param input_features: 输入特征
        :param output_features: 输出特征
        :param labels: 标签数据
        """
        feature_record = {
            'timestamp': datetime.now().isoformat(),
            'input_features': input_features.tolist() if isinstance(input_features, np.ndarray) else input_features,
            'output_features': output_features.tolist() if isinstance(output_features, np.ndarray) else output_features,
        }

        # 添加标签信息
        if labels is not None:
            feature_record['labels'] = labels

        # 添加到缓存
        self.feature_cache.append(feature_record)

        # 如果缓存达到限制，写入文件
        if len(self.feature_cache) >= self.cache_limit:
            self.flush_cache()

    def __del__(self):
        """
        在对象销毁时确保缓存被写入
        """
        self.flush_cache()
