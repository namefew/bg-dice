import numpy as np
import os
import pickle
from collections import defaultdict
from scipy.spatial import KDTree

import config
from video_processor import DiceVideoProcessor

dice_classifier = None
def get_cnn_instance():
    global dice_classifier
    if dice_classifier is None:
        dice_classifier = FeatureAnalyzer()
    return dice_classifier

class FeatureAnalyzer:
    def __init__(self, folder_path='train/features'):
        self.folder_path = folder_path
        self.load_samples()
        self.config = config.get_instance()
        # 构建 KDTree
        self.build_kdtree()

    def build_kdtree(self):
        all_points = np.array([
            [s['coordinates'][0], s['coordinates'][1]]
            for s in self.samples
            if 'coordinates' in s and len(s['coordinates']) == 2
        ])
        if all_points.size > 0:
            self.tree = KDTree(all_points)
        else:
            self.tree = None

    def _load_all_samples(self):
        """加载所有.npy文件并提取关键特征"""
        samples = []
        if os.path.exists(self.folder_path):
            for f in os.listdir(self.folder_path):
                if f.endswith('.npy'):
                    array = np.load(os.path.join(self.folder_path, f))
                    # 解析特征（与DiceDataset保持一致）
                    x = array[2]/2 + array[0]
                    y =min(array[2]/2,array[3]/2) + array[1]
                    shape_feat = (array[2], array[3])
                    next_dot = int(f.split('_')[0])  # 目标点数
                    samples.append({
                        'coordinates': (x, y),
                        'shape': shape_feat,
                        'current_dot':array[6],
                        'next_dot': next_dot
                        # 'raw_features': array
                    })
        return samples
    def save_samples(self, file_path='new-samples.pkl'):
        """保存处理后的样本数据到文件"""

        dirname = os.path.dirname(file_path)
        # 确保目录存在
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # 保存处理后的结构化数据
        with open(file_path, 'wb') as f:
            pickle.dump(self.samples, f)
        print(f"成功保存{len(self.samples)}个样本到 {file_path}")

    def load_samples(self, file_path='new-samples.pkl'):
        """从文件加载已处理的样本数据"""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.samples = pickle.load(f)
            print(f"从 {file_path} 成功加载{len(self.samples)}个样本")
        else:
            print("保存文件不存在，将重新加载原始数据")
            self.samples = self._load_all_samples()

    def find_nearby_samples(self, target_x, target_y, radius=5):
        """查找指定坐标半径范围内的样本"""
        if self.tree is None:
            return []

        indices = self.tree.query_ball_point([target_x, target_y], radius)
        return [self.samples[i] for i in indices]

    def _predict(self, target_x, target_y, current_dot, min_rate=0.2):
        state = {}
        rule_counts = defaultdict(int)
        results = {}

        # 提取所有样本的 next_dot 和 current_dot
        nearby_all = self.find_nearby_samples(target_x, target_y, max(self.config.get('radius_values', [2, 3, 4])))
        dot_nexts = defaultdict(int)
        dot_counts = defaultdict(int)
        dot_same = 0
        dot_seven = 0
        dot_other = 0

        for sample in nearby_all:
            dot_nexts[sample['next_dot']] += 1
            if sample['current_dot'] == sample['next_dot']:
                dot_same += 1
            elif sample['current_dot'] + sample['next_dot'] == 7:
                dot_seven += 1
            else:
                dot_other += 1
            if sample['current_dot'] == current_dot:
                dot_counts[sample['next_dot']] += 1

        # 遍历不同的半径范围
        radius_values = self.config.get('radius_values', [2, 3, 4])
        for r in radius_values:
            nearby = self.find_nearby_samples(target_x, target_y, r)
            if not nearby:
                continue

            total = len(nearby)
            if total < 5:
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
    # def _predict(self, target_x, target_y, current_dot, min_rate = 0.2):
    #     """多半径概率分析"""
    #     state = {}
    #     rule_counts = defaultdict(int)  # 统计各规则触发次数
    #     results = {}
    #     radius_values = self.config.get('radius_values', [2, 3, 4])
    #
    #     for r in radius_values:
    #         nearby = self.find_nearby_samples(target_x, target_y, r)
    #         if not nearby:
    #             continue
    #         # 统计点数出现频率
    #         dot_nexts = defaultdict(int)
    #         dot_counts = defaultdict(int)
    #         dot_same = 0
    #         dot_seven = 0
    #         dot_other = 0
    #         for sample in nearby:
    #             dot_nexts[sample['next_dot']] += 1
    #             if sample['current_dot'] == sample['next_dot']:
    #                 dot_same += 1
    #             elif sample['current_dot'] + sample['next_dot'] ==7:
    #                 dot_seven += 1
    #             else:
    #                 dot_other += 1
    #             if sample['current_dot'] == current_dot:
    #                 dot_counts[sample['next_dot']] += 1
    #         # 计算概率
    #         total = len(nearby)
    #         if total>5:
    #
    #             next_probs = {k: v / total for k, v in dot_nexts.items()}
    #             most_dot = max(next_probs, key=next_probs.get)
    #             # 修改为带判空保护的版本：
    #             if dot_counts:
    #                 current_most_dot = max(dot_counts, key=lambda k: dot_counts[k])
    #                 current_most_prob = dot_counts[current_most_dot]
    #             else:
    #                 current_most_dot = None
    #                 current_most_prob = 0
    #             state[r] = {
    #                 'total': total,
    #                 'same_prob':dot_same/total, # 相同点数出现的概率
    #                 'seven_prob':dot_seven/total,   # 点数之和为7出现的概率
    #                 'other_prob':dot_other/total,   # 其他点数出现的概率
    #                 'current_total':sum(dot_counts), # 当前点数出现的次数
    #                 'current_most_dot': current_most_dot, # 当前点数出现时下次出现次数最多的点数
    #                 'current_most_prob':current_most_prob, # 当前点数出现时下次出现次数最多的点数出现的概率
    #                 'next_most_dot': most_dot,  # 下次出现次数最多的点数
    #                 'next_most_prob':next_probs.get(most_dot), # 下次出现次数最多的点数的概率
    #             }
    #             final_rule = None
    #             if state[r]['same_prob'] > min_rate and state[r]['same_prob'] >= state[r]['seven_prob']:
    #                 final_rule = {'next': current_dot, 'prob': state[r]['same_prob'],
    #                               'sample': state[r]['total'], 'rule': 1,'radius':r}
    #             elif state[r]['seven_prob'] > min_rate and state[r]['seven_prob'] >= state[r]['same_prob']:
    #                 final_rule = {'next': 7 - current_dot, 'prob': state[r]['seven_prob'],
    #                               'sample': state[r]['total'], 'rule': 2,'radius':r}
    #             # elif state[r]['current_most_prob'] > min_rate and state[r]['current_most_prob']>=state[r]['next_most_prob']:
    #             #     final_rule = {'next': state[r]['current_most_dot'],
    #             #                   'prob': state[r]['current_most_prob'] / state[r]['current_total'],
    #             #                   'sample': state[r]['current_total'], 'rule': 3,'radius':r}
    #             # elif state[r]['next_most_prob'] > min_rate and state[r]['next_most_prob'] >= state[r]['current_most_prob']:
    #             #     final_rule = {'next': state[r]['next_most_dot'],
    #             #                   'prob': state[r]['next_most_prob'],
    #             #                   'sample': state[r]['total'], 'rule': 4,'radius':r}
    #
    #             if final_rule:
    #                 print(final_rule)
    #                 results[r] = final_rule
    #                 rule_counts[final_rule['next']] += 1
    #
    #     # 选择出现次数最多的预测结果
    #
    #     if rule_counts:
    #         max_count = max(rule_counts.values())
    #         if max_count == 1:
    #             all_candidates = [r for r in results.values()]
    #             sorted_by_rule = sorted(all_candidates, key=lambda x: x['rule'])
    #             return sorted_by_rule[0] if sorted_by_rule else None
    #         else:
    #             candidates = [k for k, v in rule_counts.items() if v == max_count]
    #             best_next = max(
    #                 candidates,
    #                 key=lambda x: max(r['prob'] for r in results.values() if r['next'] == x)
    #             )
    #             # 获取概率最高的结果
    #             best_result = max(
    #                 (r for r in results.values() if r['next'] == best_next),
    #                 key=lambda x: x['prob']
    #             )
    #             return best_result
    #     return None

    def predict(self, frame:np.ndarray, background):
        video_processor = DiceVideoProcessor(background)
        features = video_processor.detect_dice_feature(frame)
        if features is None:
            return None, None
        x = features[2]/2 + features[0]  # 根据实际特征位置调整索引
        y = min(features[3]/2,features[2]/2) + features[1]
        result = self._predict(x,y,int(features[6]))
        if result:
            return result['next'], result['prob']
        else:
            return None,None
    def predict_image_top(self, frame: np.ndarray, background, n=6,angle_diff=0):
        if background is None:
            return [], []
        next, prob = self.predict(frame, background)
        if next:
            # 计算其他点数的概率
            other_prob = (1 - prob) / 5
            # 创建前N个预测结果及其概率
            topN_class = [next] + [i for i in range(1, 7) if i != next][:n - 1]
            topN_prob = [prob] + [other_prob] * (n - 1)
            return topN_class, topN_prob
        else:
            return [], []

    def add_sample(self,current_dot, last_frame, background):
        if background is None:
            return [], []
        video_processor = DiceVideoProcessor(background)
        features = video_processor.detect_dice_feature(last_frame)
        if features is None:
            return [], []
        x = features[2]/2 + features[0]  # 根据实际特征位置调整索引
        y = min(features[3]/2,features[2]/2) + features[1]
        shape_feat = (features[2], features[3])
        self.samples.append({
            'coordinates': (x, y),
            'shape': shape_feat,
            'current_dot': features[6],
            'next_dot': current_dot
        })
        self.build_kdtree()
        self.save_samples()

# 使用示例
if __name__ == "__main__":
    analyzer = FeatureAnalyzer('train/features')
    # 初始化分析器时自动保存
    # analyzer = FeatureAnalyzer('train/features')
    analyzer.save_samples()  # 保存到默认路径saved_samples.pkl

    # 后续使用可以直接加载
    analyzer.load_samples()  # 从保存文件快速加载
    # 示例坐标（替换为实际需要分析的坐标）
    for x in range(20, 220, 4):
        for y in range(20, 220, 4):
            result = analyzer._predict(x, y,1)
            if result:
                print(f"预测点数: {result['next']} (置信度: {result['prob']:.2%}, 样本数: {result['sample']}, 规则: {result['rule']})")
            else:
                print("该区域未找到历史数据")
    # target_x, target_y = 112, 112  # 图像中心位置
    # #
    # result = analyzer.predict(target_x, target_y,6)
    # if result:
    #     print(f"预测点数: {result['next']} (置信度: {result['prob']:.2%}, 样本数: {result['sample']}, 规则: {result['rule']})")
    # else:
    #     print("该区域未找到历史数据")
