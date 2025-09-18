from collections import defaultdict

import numpy as np

import config
import train_resnet
from dice_classifier_1 import FeatureAnalyzer
from video_processor import DiceVideoProcessor


class BigOddClassifier(FeatureAnalyzer):
    def __init__(self, folder_path='train/features'):
        super().__init__(folder_path)


    def predict_image_top_0(self, frame: np.ndarray, background, n=6, angle_diff=0):
        if background is None:
            return [], []
        video_processor = DiceVideoProcessor(background)
        features = video_processor.extract_simple_feature(frame)
        if features is None:
            return [], []
        x = features[2] / 2 + features[0]  # 根据实际特征位置调整索引
        y = features[3] / 2 + features[1]
        result = self._predict(x, y, int(features[4]), angle_diff=angle_diff,min_rate=config.get_instance().get('min_rate',0.6))
        big_prob = result['big']*0.05+0.5
        odd_prob = result['odd']*0.05+0.5

        prob_map = {
            1: (1 - big_prob) * odd_prob / 2,  # 小且单（1,3均分）
            2: (1 - big_prob) * (1 - odd_prob),  # 小且双（仅2）
            3: (1 - big_prob) * odd_prob / 2,  # 小且单（1,3均分）
            4: big_prob * (1 - odd_prob) / 2,  # 大且双（4,6均分）
            5: big_prob * odd_prob,  # 大且单（仅5）
            6: big_prob * (1 - odd_prob) / 2  # 大且双（4,6均分）
        }
        # 归一化处理
        total = sum(prob_map.values())
        prob_map = {k: v / total for k, v in prob_map.items()}
        # 将字典转换为有序的数值列表
        sorted_items = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
        sorted_numbers = [item[0] for item in sorted_items]
        sorted_probs = [item[1] for item in sorted_items]
        return sorted_numbers, sorted_probs

    def predict_image_top(self, frame: np.ndarray, background, n=6, angle_diff=0):
        dot, conf = train_resnet.get_cnn_instance().predict_image(frame)
        if dot == 0:
            return [], []
        elif dot == 1 or dot == 3:  # 大双
            return [4, 6, 2, 5, 1, 3], [0.3, 0.3, 0.2, 0.2, 0, 0]
        elif dot == 2:  # 大单
            return [5, 1, 3, 4, 6, 2], [0.5, 0.125, 0.125, 0.125, 0.125, 0]
        elif dot == 4 or dot == 6:  # 小单
            return [1, 3, 5, 2, 4, 6], [0.3, 0.3, 0.2, 0.2, 0, 0]
        elif dot == 5:  # 小双
            return [2, 4, 6, 1, 3, 5], [0.5, 0.125, 0.125, 0.125, 0.125, 0]
        else:
            return [], []
    def _predict(self, target_x, target_y, current_dot, angle_diff=0, min_rate=0.6):
        state = {}
        rule_counts = defaultdict(int)
        rule_counts['big']=0
        rule_counts['odd']=0
        # 从配置获取参数，默认值保障基础运行
        radius_values = self.config.get('radius_values', [2, 3, 4])

        for r in radius_values:
            nearby = self.find_nearby_samples(target_x, target_y, r, angle_diff)
            if not nearby or len(nearby) < 3:
                continue

            # 初始化统计量
            dot_big = dot_odd = big_to_big = odd_to_odd = small_to_small = even_to_even = 0

            for sample in nearby:
                next_dot = int(sample[-1])
                current = int(sample[4])  # 使用样本中的当前点数

                # 大小统计
                if next_dot > 3:
                    dot_big += 1
                    if current > 3:
                        big_to_big += 1
                elif current <= 3:  # 仅当当前是小点时统计小->小
                    small_to_small += 1

                # 单双统计
                if next_dot % 2 == 1:
                    dot_odd += 1
                    if current % 2 == 1:
                        odd_to_odd += 1
                elif current % 2 == 0:  # 仅当当前是双点时统计双->双
                    even_to_even += 1

            # 计算概率（带防零处理）
            total = len(nearby)
            safe_div = lambda x, y: x / y if y != 0 else 0.0

            state[r] = {
                'total': total,
                'big_prob': safe_div(dot_big, total),
                'odd_prob': safe_div(dot_odd, total),
                'big_to_big': safe_div(big_to_big, dot_big),
                'small_to_small': safe_div(small_to_small, (total - dot_big)),
                'odd_to_odd': safe_div(odd_to_odd, dot_odd),
                'even_to_even': safe_div(even_to_even, (total - dot_odd))
            }
            print(state[r])
            # 规则判断
            if state[r]['big_prob'] >= min_rate:
                rule_counts['big'] += 1
            elif state[r]['big_prob']<1-min_rate:
                rule_counts['big'] -= 1
            if state[r]['odd_prob'] >= min_rate:
                rule_counts['odd'] += 1
            if state[r]['odd_prob'] <1 - min_rate:
                rule_counts['odd'] -= 1
            if current>3:
                if state[r]['big_to_big'] >= min_rate:
                    rule_counts['big'] +=1
                elif state[r]['big_to_big'] <1-min_rate:
                    rule_counts['big'] -=1
            else:
                if state[r]['small_to_small'] >= min_rate:
                    rule_counts['big'] -=1
                elif state[r]['small_to_small'] <1-min_rate:
                    rule_counts['big'] +=1
            if current%2==1:
                if state[r]['odd_to_odd'] >= min_rate:
                    rule_counts['odd'] +=1
                elif state[r]['odd_to_odd'] <1-min_rate:
                    rule_counts['odd'] -=1
            else:
                if state[r]['even_to_even'] >= min_rate:
                    rule_counts['odd'] -=1
                elif state[r]['even_to_even'] <1-min_rate:
                    rule_counts['odd'] +=1

        return rule_counts

if __name__ == "__main__":
    analyzer = BigOddClassifier()
    analyzer._predict(112, 112, 1, 0)
    analyzer._predict(112, 112, 1, -1)
    analyzer._predict(112, 112, 1, 1)
