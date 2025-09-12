import os
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
import sqlite3
from datetime import datetime
from pathlib import Path
import atexit
import threading

import config
from video_processor import DiceVideoProcessor


class FeatureDatabase:
    _local = threading.local()  # 线程本地存储
    _instances = []  # 跟踪所有实例
    _lock = threading.Lock()  # 用于保护实例列表的锁

    def __init__(self, db_path='data/dice_features.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 不在初始化时创建连接，而是在需要时创建
        self.conn = None

        # 注册实例以便在退出时清理
        with FeatureDatabase._lock:
            FeatureDatabase._instances.append(self)

        atexit.register(self._cleanup)

    def _get_conn(self):
        """为每个线程创建独立的连接"""
        if not hasattr(FeatureDatabase._local, 'conn'):
            FeatureDatabase._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,  # 禁用线程检查
                timeout=30,  # 增加超时时间
                isolation_level=None  # 启用自动提交模式
            )
            # 启用WAL模式提升并发性能
            FeatureDatabase._local.conn.execute('PRAGMA journal_mode=WAL;')
            # 设置锁定模式
            FeatureDatabase._local.conn.execute('PRAGMA locking_mode=NORMAL;')
        return FeatureDatabase._local.conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript('''
               CREATE TABLE IF NOT EXISTS dice_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    width REAL NOT NULL,
                    height REAL NOT NULL,
                    current_dot INTEGER NOT NULL,
                    other_features BLOB,
                    next_dot INTEGER NOT NULL,
                    angle_diff REAL NOT NULL,
                    sample_type TEXT CHECK(sample_type IN ('zero','pos','neg')),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
               CREATE INDEX IF NOT EXISTS idx_timestamp ON dice_features(timestamp);
               ''')
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"初始化数据库失败: {str(e)}")

    def insert_feature(self, feature_data):
        insert_sql = '''
           INSERT INTO dice_features 
           (x, y, width, height, current_dot, other_features, next_dot, angle_diff, sample_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           '''
        conn = self._get_conn()
        try:
            conn.execute('BEGIN IMMEDIATE')  # 显式开始事务
            conn.execute(insert_sql, feature_data)
            conn.execute('COMMIT')  # 提交事务
        except sqlite3.IntegrityError as e:
            conn.execute('ROLLBACK')  # 回滚事务
            print(f"数据完整性错误: {str(e)}")
        except sqlite3.OperationalError as e:
            conn.execute('ROLLBACK')  # 回滚事务
            print(f"数据库操作失败: {str(e)}")

    def close_all(self):
        """关闭所有线程的连接"""
        if hasattr(FeatureDatabase._local, 'conn'):
            try:
                FeatureDatabase._local.conn.close()
            except:
                pass
            delattr(FeatureDatabase._local, 'conn')

    def close(self):
        """关闭当前实例的连接"""
        self.close_all()

    def _cleanup(self):
        """清理资源"""
        self.close_all()

    def get_features_by_time(self, start_time: datetime, end_time: datetime):
        query_sql = '''
           SELECT * FROM dice_features 
           WHERE timestamp BETWEEN ? AND ?
           ORDER BY timestamp DESC
           '''
        conn = self._get_conn()
        cursor = conn.execute(query_sql, (start_time.isoformat(), end_time.isoformat()))
        return cursor.fetchall()

    def get_weekly_patterns(self):
        query_sql = '''
           SELECT 
               strftime('%Y-%W', timestamp) as week,
               sample_type,
               current_dot,
               next_dot,
               COUNT(*) as count
           FROM dice_features
           GROUP BY week, sample_type, current_dot, next_dot
           '''
        conn = self._get_conn()
        cursor = conn.execute(query_sql)
        return cursor.fetchall()

    @classmethod
    def close_all_instances(cls):
        """关闭所有实例的连接"""
        with cls._lock:
            for instance in cls._instances:
                instance.close()
            cls._instances.clear()


# 确保在程序退出时关闭所有数据库连接
atexit.register(FeatureDatabase.close_all_instances)

dice_classifier = None

def get_cnn_instance():
    global dice_classifier
    if dice_classifier is None:
        dice_classifier = FeatureAnalyzer()
    return dice_classifier

"""
特征数组各列定义：
0: x坐标
1: y坐标 
2: 宽度w
3: 高度h
4: current_dot (当前点数)
5: 其他特征
6: next_dot (下一个点数)
"""
class FeatureAnalyzer:
    def __init__(self, folder_path='train/features'):
        self.folder_path = folder_path
        self.zero,self.pos,self.neg = self.load_combined_features()
        self.config = config.get_instance()
        # 构建 KDTree
        # 修正后（使用中心坐标）
        zero_points = np.column_stack((
            self.zero[:, 0] + self.zero[:, 2] / 2,  # x_center = x + w/2
            self.zero[:, 1] + self.zero[:, 3] / 2  # y_center = y + h/2
        )) if self.zero.size > 0 else np.array([])

        pos_points = np.column_stack((
            self.pos[:, 0] + self.pos[:, 2] / 2,
            self.pos[:, 1] + self.pos[:, 3] / 2
        )) if self.pos.size > 0 else np.array([])

        neg_points = np.column_stack((
            self.neg[:, 0] + self.neg[:, 2] / 2,
            self.neg[:, 1] + self.neg[:, 3] / 2
        )) if self.neg.size > 0 else np.array([])
        if zero_points.size > 0:
            self.zero_tree = KDTree(zero_points)
        else:
            self.zero_tree = None
        if pos_points.size > 0:
            self.pos_tree = KDTree(pos_points)
        else:
            self.pos_tree = None
        if neg_points.size > 0:
            self.neg_tree = KDTree(neg_points)
        else:
            self.neg_tree = None
        self.all = np.concatenate([arr for arr in [self.zero, self.pos, self.neg]])
        all_points = np.column_stack((
            self.all[:, 0] + self.all[:, 2] / 2,
            self.all[:, 1] + self.all[:, 3] / 2
        )) if self.neg.size > 0 else np.array([])
        self.all_tree = KDTree(all_points)
        self.db = FeatureDatabase()
    def is_force_add_sample(self):
        return self.config.get('force_add_sample', False)
    def load_features_by_prefix(self,folder_path='features'):
        # 初始化三个列表暂存不同类别的特征
        zero_list = []
        negative_list = []
        positive_list = []
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(folder_path, filename)

                    # 提取前缀数字（基于你的原始实现）
                    try:
                        num = int(filename.split('_')[0])
                        feature = np.load(file_path)

                        if num == 0:
                            zero_list.append(feature)
                        elif num < 0:
                            negative_list.append(feature)
                        else:
                            positive_list.append(feature)
                    except (ValueError, IndexError):
                        continue  # 忽略不符合命名规范的文件

        # 将列表转换为numpy数组（按行堆叠）
        return (
            np.vstack(zero_list) if zero_list else np.array([]),
            np.vstack(positive_list) if positive_list else np.array([]),
            np.vstack(negative_list) if negative_list else np.array([])

        )

    def save_combined_features(self,zero_arr, pos_arr, neg_arr, save_path='features_combined.npz'):
        """
        将三个特征数组合并保存到单个压缩文件中
        :param zero_arr: 0前缀特征数组
        :param neg_arr: 负数前缀特征数组
        :param pos_arr: 正数前缀特征数组
        :param save_path: 保存路径，默认当前目录
        """
        np.savez_compressed(
            save_path,
            zero_features=zero_arr,
            positive_features=pos_arr,
            negative_features=neg_arr
        )

    def load_combined_features(self,file_path='features_combined.npz'):
        """ 从npz文件加载所有特征 """
        if not os.path.exists(file_path):
            return self.load_features_by_prefix()
        data = np.load(file_path)
        return (
            data['zero_features'],
            data['positive_features'],
            data['negative_features']

        )



    # 修改后（适配numpy数组结构）
    def plot_all_label_heatmaps(self, data):
        """ label_type: 'zero'/'pos'/'neg' 对应不同类别 """
        labels = [0, 1, 2]
        fig, axs = plt.subplots(1, len(labels), figsize=(30, 8))

        for i, label in enumerate(labels):
            x_coords = []
            y_coords = []

            for j in range(len(data)):
                d = data[j]
                current_dot = d[4]
                next_dot = d[-1]
                if label == 1 and current_dot==next_dot or label == 2 and current_dot+next_dot==7 or label==0 and current_dot!=next_dot and current_dot+next_dot!=7:
                    x = d[0] + d[2] / 2
                    y = d[1] + d[3] / 2
                    x_coords.append(x)
                    y_coords.append(y)

            # 绘制热力图
            hb = axs[i].hexbin(x_coords, y_coords,
                               gridsize=int(112),
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

    def get_heatmap_data(data):
        labels = [0, 1, 2]
        heatmap_data = []

        for label in labels:
            x_coords = []
            y_coords = []

            for j in range(len(data)):
                sample = data[j]
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


    def find_nearby_samples(self, target_x, target_y, radius=5, angle_diff=0):
        """查找指定坐标半径范围内的样本"""
        the_tree = self.zero_tree
        samples  = self.zero
        if angle_diff<=-1:
            the_tree = self.neg_tree
            samples = self.neg
        elif angle_diff>=1:
            the_tree = self.pos_tree
            samples = self.pos

        if config.get_instance().get('use_all_samples', True):
            the_tree = self.all_tree
            samples = self.all
        elif config.get_instance().get('use_sample_index',0)==0:
            the_tree = self.zero_tree
            samples = self.zero
        elif config.get_instance().get('use_sample_index',0)==1:
            the_tree = self.pos_tree
            samples = self.pos
        elif config.get_instance().get('use_sample_index',0)==-1:
            the_tree = self.neg_tree
            samples = self.neg
        if the_tree is None:
            return []
        indices = the_tree.query_ball_point([target_x, target_y], radius)
        return [samples[i] for i in indices]

    def _predict(self, target_x, target_y, current_dot, angle_diff=0, min_rate=0.2):
        state = {}
        rule_counts = defaultdict(int)
        results = {}
        # 遍历不同的半径范围
        radius_values = self.config.get('radius_values', [2, 3, 4])
        for r in radius_values:
            # 修正后的循环结构示意
            nearby = self.find_nearby_samples(target_x, target_y, r, angle_diff)
            if not nearby:
                continue

            # 重新初始化当前半径的统计量
            dot_nexts = defaultdict(int)
            dot_same = dot_seven = dot_other = 0
            dot_counts = defaultdict(int)

            for sample in nearby:
                next_dot = sample[-1]
                current_sample_dot = int(sample[4])  # 类型转换

                dot_nexts[next_dot] += 1
                if current_sample_dot == next_dot:
                    dot_same += 1
                elif current_sample_dot + next_dot == 7:
                    dot_seven += 1
                else:
                    dot_other += 1

                if current_sample_dot == current_dot:
                    dot_counts[next_dot] += 1

            total = len(nearby)
            if total < 3:
                continue

            # 计算概率
            next_probs = {k: v / total for k, v in dot_nexts.items()}
            most_dot = max(next_probs, key=next_probs.get)

            # 当前点数出现时下次出现次数最多的点数
            current_most_dot = max(dot_counts, key=lambda k: dot_counts[k]) if dot_counts else None
            current_most_prob = dot_counts[current_most_dot] if current_most_dot else 0

            state[r] = {
                'total': total,
                'same_prob': dot_same / total,
                'seven_prob': dot_seven / total,
                'other_prob': dot_other / total,
                'current_total': sum(dot_counts),
                'current_most_dot': current_most_dot,
                'current_most_prob': current_most_prob,
                'next_most_dot': most_dot,
                'next_most_prob': next_probs.get(most_dot),
            }

            final_rule = None
            if state[r]['same_prob'] > min_rate and state[r]['same_prob'] >= state[r]['seven_prob']:
                final_rule = {
                    'next': current_dot,
                    'prob': state[r]['same_prob'],
                    'sample': state[r]['total'],
                    'rule': 1,
                    'radius': r,
                }
            elif state[r]['seven_prob'] > min_rate and state[r]['seven_prob'] >= state[r]['same_prob']:
                final_rule = {
                    'next': 7 - current_dot,
                    'prob': state[r]['seven_prob'],
                    'sample': state[r]['total'],
                    'rule': 2,
                    'radius': r,
                }

            if final_rule:
                print(final_rule)
                results[r] = final_rule
                rule_counts[final_rule['next']] += 1

        # 选择出现次数最多的预测结果
        if rule_counts:
            max_count = max(rule_counts.values())
            if max_count == 1:
                all_candidates = [r for r in results.values()]
                sorted_by_rule = sorted(all_candidates, key=lambda x: x['rule'])
                return sorted_by_rule[0] if sorted_by_rule else None
            else:
                candidates = [k for k, v in rule_counts.items() if v == max_count]
                best_next = max(
                    candidates,
                    key=lambda x: max(r['prob'] for r in results.values() if r['next'] == x)
                )
                best_result = max(
                    (r for r in results.values() if r['next'] == best_next),
                    key=lambda x: x['prob']
                )
                return best_result
        return None

    def predict(self, frame: np.ndarray, background,angle_diff=0):
        video_processor = DiceVideoProcessor(background)
        features = video_processor.extract_simple_feature(frame)
        if features is None:
            return None, None
        x = features[2] / 2 + features[0]  # 根据实际特征位置调整索引
        y = features[3] / 2 + features[1]
        result = self._predict(x, y, int(features[4]),angle_diff=angle_diff, min_rate=config.get_instance().get('single_min_rate',0.2))
        if result:
            return result['next'], result['prob']
        else:
            return None, None

    def predict_image_top(self, frame: np.ndarray, background, n=6, angle_diff=0):
        if background is None:
            return [], []
        next, prob = self.predict(frame, background,angle_diff=angle_diff)
        if next:
            # 计算其他点数的概率
            other_prob = (1 - prob) / 5
            # 创建前N个预测结果及其概率
            topN_class = [next] + [i for i in range(1, 7) if i != next][:n - 1]
            topN_prob = [prob] + [other_prob] * (n - 1)
            return topN_class, topN_prob
        else:
            return [], []

    def add_sample(self, current_dot, last_frame, background, angle_diff=0):
        if background is None:
            return
        video_processor = DiceVideoProcessor(background)
        features = video_processor.detect_dice_feature(last_frame)
        if features is None:
            return

        # 确定样本类别
        sample_type = 'zero'
        if angle_diff <= -1:
            sample_type = 'neg'
        elif angle_diff >= 1:
            sample_type = 'pos'

        # 构造新样本特征数组
        new_sample = np.array([
            features[0],  # x
            features[1],  # y
            features[2],  # w
            features[3],  # h
            features[6],  # current_dot（根据特征定义第6列）
            features[5],  # 其他特征
            current_dot  # next_dot
        ])
        # 新增数据库存储
        db_data = (
            features[0],  # x
            features[1],  # y
            features[2],  # width
            features[3],  # height
            int(features[6]),  # current_dot
            features[5].tobytes(),  # 序列化其他特征
            current_dot,  # next_dot
            angle_diff,
            sample_type
        )
        self.db.insert_feature(db_data)
        # 更新特征数组
        target_array = getattr(self, sample_type)
        if target_array.size == 0:
            target_array = new_sample.reshape(1, -1)
        else:
            target_array = np.vstack([target_array, new_sample])
        setattr(self, sample_type, target_array)

        # 重新构建对应类别的 KDTree
        points = np.column_stack((
            target_array[:, 0] + target_array[:, 2] / 2,
            target_array[:, 1] + target_array[:, 3] / 2
        ))
        tree_name = f"{sample_type}_tree"
        setattr(self, tree_name, KDTree(points) if points.size > 0 else None)

        if self.all.size == 0:
            self.all = new_sample.reshape(1, -1)  # 空数组时初始化二维结构
        else:
            self.all = np.vstack([self.all, new_sample])  # 非空时垂直堆叠
        all_points = np.column_stack((
            self.all[:, 0] + self.all[:, 2] / 2,
            self.all[:, 1] + self.all[:, 3] / 2
        ))
        self.all_tree = KDTree(all_points)
        # 保存更新后的特征数据
        self.save_combined_features(self.zero, self.pos, self.neg)


if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    analyzer.save_combined_features(analyzer.zero, analyzer.pos, analyzer.neg)
    # analyzer.plot_all_label_heatmaps(analyzer.zero)
    # analyzer.plot_all_label_heatmaps(analyzer.pos)
    # analyzer.plot_all_label_heatmaps(analyzer.neg)
    # dataset.plot_all_label_heatmaps_to_table()

    analyzer._predict(112, 112, 1, 0)
    analyzer._predict(112, 112, 1, -1)
    analyzer._predict(112, 112, 1, 1)
    # dataset.plot_dot_distribution()
    # classifier = DiceClassifier()
    # classifier.train(folder_path='train/features')

