# -*- coding: utf-8 -*-
"""
骰子点数预测模型
基于18万条有序点数序列，利用每周换骰规律和序列依赖建模
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_data(file_path):
    """加载文本数据，每行或连续字符为1-6的数字"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    rolls = [int(char) for char in data if char.isdigit()]
    return np.array(rolls)


def eda_frequency(rolls):
    """全局频率分布分析"""
    counts = Counter(rolls)
    freq = {k: counts[k] / len(rolls) for k in range(1, 7)}

    print("各点数频率分布:")
    for k, v in freq.items():
        print(f"点数 {k}: {v:.3f} ({v * 100:.1f}%)")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(freq.keys()), y=list(freq.values()), palette='Blues_d')
    plt.title('骰子点数频率分布（总计 {:,} 局）'.format(len(rolls)), fontsize=16)
    plt.xlabel('点数')
    plt.ylabel('频率')
    plt.axhline(y=1 / 6, color='r', linestyle='--', label='理论均匀值 (16.7%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('frequency_distribution.png', dpi=200)
    plt.show()


def eda_autocorrelation(rolls, lag_max=50):
    """自相关分析"""
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(rolls, nlags=lag_max)

    plt.figure(figsize=(12, 6))
    plt.stem(range(lag_max + 1), acf_vals)  # 移除use_line_collection参数
    plt.title('点数序列自相关图（前{}阶）'.format(lag_max), fontsize=16)
    plt.xlabel('滞后阶数')
    plt.ylabel('自相关系数')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('autocorrelation.png', dpi=200)
    plt.show()

    print(f"滞后1阶自相关: {acf_vals[1]:.3f}")
    print(f"滞后2阶自相关: {acf_vals[2]:.3f}")


def eda_transition_matrix(rolls):
    """状态转移矩阵分析"""
    trans_matrix = np.zeros((6, 6))
    for i in range(len(rolls) - 1):
        curr = rolls[i] - 1
        next_ = rolls[i + 1] - 1
        trans_matrix[curr, next_] += 1
    trans_matrix = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(trans_matrix, annot=True, fmt=".3f", cmap='Blues',
                xticklabels=[f'下一局={i + 1}' for i in range(6)],
                yticklabels=[f'当前={i + 1}' for i in range(6)])
    plt.title('状态转移概率矩阵（Markov 模型）', fontsize=16)
    plt.tight_layout()
    plt.savefig('transition_matrix.png', dpi=200)
    plt.show()

    print("\n显著高于16.7%的转移概率:")
    found = False
    for i in range(6):
        for j in range(6):
            if trans_matrix[i, j] > 0.20:
                print(f"P(下一局={j + 1} | 当前={i + 1}) = {trans_matrix[i, j]:.3f}")
                found = True
    if not found:
        print("无显著高概率转移（>20%）")


def eda_weekly_pattern(rolls, games_per_week=14000, window_size=10):
    """分析每周换骰后的前N局模式"""
    first_n_after_change = []
    for week_start in range(0, len(rolls), games_per_week):
        end_idx = min(week_start + window_size, len(rolls))
        first_n_after_change.extend(rolls[week_start:end_idx])

    first_n_counts = Counter(first_n_after_change)
    first_n_freq = {k: first_n_counts[k] / len(first_n_after_change) for k in range(1, 7)}

    print(f"\n换骰后前 {window_size} 局的频率:")
    for k, v in first_n_freq.items():
        print(f"点数 {k}: {v:.3f} ({v * 100:.1f}%)")

    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    freq_all = {k: Counter(rolls)[k] / len(rolls) for k in range(1, 7)}

    ax[0].bar(freq_all.keys(), freq_all.values(), alpha=0.7, color='blue')
    ax[0].axhline(y=1 / 6, color='r', linestyle='--')
    ax[0].set_title('全局频率分布')
    ax[0].set_ylabel('频率')

    ax[1].bar(first_n_freq.keys(), first_n_freq.values(), alpha=0.7, color='green')
    ax[1].axhline(y=1 / 6, color='r', linestyle='--')
    ax[1].set_title(f'换骰后前 {window_size} 局频率')
    ax[1].set_ylabel('频率')

    plt.tight_layout()
    plt.savefig('weekly_pattern.png', dpi=200)
    plt.show()

def create_features(rolls, lookback=10, games_per_week=14000):
    """构造增强特征：历史序列 + 统计 + 时间周期 + 状态转移"""
    X, y = [], []
    for i in range(lookback, len(rolls)):
        past = rolls[i - lookback:i]
        features = list(past)

        # 基础统计
        features.append(np.mean(past))
        features.append(np.std(past))
        features.append(np.argmax(np.bincount(past)))  # 众数

        # 移动窗口统计（过去5局）
        if len(past) >= 5:
            recent = past[-5:]
            features.append(np.mean(recent))
            features.append(np.std(recent))

        # 周期性特征
        game_id = i
        features.append(game_id % games_per_week)  # 周周期
        features.append(game_id % 2000)  # 日周期
        features.append((game_id // games_per_week) % 7)  # 星期几（假设每周换骰）

        # 状态转移特征
        if i > 0:
            prev_roll = rolls[i - 1]
            features.append(1 if prev_roll % 2 == 0 else 0)  # 奇偶性
            features.append(1 if prev_roll >= 4 else 0)     # 大小（>=4为大）

        X.append(features)
        y.append(rolls[i])
    return np.array(X), np.array(y)


def train_model(X, y):
    """训练 LightGBM 模型，增强鲁棒性和性能"""
    print(f"训练数据形状: X={X.shape}, y={y.shape}")
    print(f"标签值范围: [{y.min()}, {y.max()}]")
    print(f"标签值分布: {Counter(y)}")

    # 过滤无效标签
    valid_indices = (y >= 1) & (y <= 6)
    X_filtered = X[valid_indices]
    y_filtered = y[valid_indices]

    print(f"过滤后数据形状: X={X_filtered.shape}, y={y_filtered.shape}")

    if len(X_filtered) == 0:
        raise ValueError("没有有效的训练数据")

    split_idx = int(len(X_filtered) * 0.9)
    X_train, X_test = X_filtered[:split_idx], X_filtered[split_idx:]
    y_train, y_test = y_filtered[:split_idx], y_filtered[split_idx:]

    # 标签转换为0-based
    train_data = lgb.Dataset(X_train, label=y_train - 1)
    test_data = lgb.Dataset(X_test, label=y_test - 1, reference=train_data)

    params = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,  # L1正则化
        'lambda_l2': 0.1,  # L2正则化
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50)
        ]
    )

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) + 1
    acc = accuracy_score(y_test, y_pred_classes)

    print(f"\n✅ 测试集准确率: {acc:.3f} ({acc * 100:.1f}%)")
    print(f"🎲 随机猜测准确率: {1 / 6:.3f} (16.7%)")
    print(f"📈 相对提升: {acc - 1 / 6:.3f}")

    print("\n📋 分类报告:")
    print(classification_report(y_test, y_pred_classes))

    # 特征重要性
    lgb.plot_importance(model, max_num_features=10, figsize=(10, 6))
    plt.title('特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=200)
    plt.show()

    return model

def predict_next_roll(last_10_rolls, model, game_id, games_per_week=14000):
    """
    预测下一次点数
    last_10_rolls: 最近10次点数 [int]
    model: 训练好的 LightGBM 模型
    game_id: 当前局数（从0开始）
    """
    past = np.array(last_10_rolls)
    features = list(past)
    
    # 基础统计
    features.append(np.mean(past))
    features.append(np.std(past))
    features.append(np.argmax(np.bincount(past)))  # 众数

    # 移动窗口统计（过去5局）
    if len(past) >= 5:
        recent = past[-5:]
        features.append(np.mean(recent))
        features.append(np.std(recent))

    # 周期性特征
    features.append(game_id % games_per_week)  # 周周期
    features.append(game_id % 2000)  # 日周期
    features.append((game_id // games_per_week) % 7)  # 星期几（假设每周换骰）

    # 状态转移特征
    if len(past) > 0:
        prev_roll = past[-1]
        features.append(1 if prev_roll % 2 == 0 else 0)  # 奇偶性
        features.append(1 if prev_roll >= 4 else 0)     # 大小（>=4为大）

    X = np.array([features])
    pred_proba = model.predict(X)[0]
    pred_class = np.argmax(pred_proba) + 1  # 将预测结果从[0,5]转换回[1,6]
    confidence = np.max(pred_proba)

    return pred_class, confidence, pred_proba


def predict_next_roll_rule_based(last_rolls, game_id, games_per_week=14000):
    """
    基于规则的预测方法
    根据EDA分析结果，换骰后前几局有特定模式
    """
    # 计算当前是本周的第几局
    week_game_index = game_id % games_per_week
    
    # 如果是换骰后的前10局，使用特定规则
    if week_game_index < 10:
        # 根据EDA结果，换骰后点数6的概率较高(20.8%)
        # 点数2和5的概率也相对较高
        if week_game_index == 0:
            # 换骰后第一局，预测点数6
            return 6, 0.208, [0.138, 0.185, 0.138, 0.162, 0.169, 0.208]
        elif week_game_index < 3:
            # 换骰后前3局，仍然偏向预测6
            return 6, 0.20, [0.15, 0.17, 0.15, 0.16, 0.16, 0.20]
        else:
            # 换骰后第3-10局，使用更平衡的预测
            return 2, 0.185, [0.138, 0.185, 0.138, 0.162, 0.169, 0.208]
    
    # 其他情况下，使用模型预测
    # 由于模型性能不佳，我们返回均匀分布
    uniform_proba = [1/6] * 6
    return np.random.choice([1, 2, 3, 4, 5, 6]), 1/6, uniform_proba


def main():
    file_path = 'bg_history_result.txt'  # 确保文件在同一目录

    print("🎲 开始加载数据...")
    rolls = load_data(file_path)
    print(f"✅ 数据加载完成，共 {len(rolls):,} 条记录\n")

    print("📊 正在进行探索性数据分析...")
    eda_frequency(rolls)
    eda_autocorrelation(rolls)
    eda_transition_matrix(rolls)
    eda_weekly_pattern(rolls)

    print("\n🧠 正在构造特征并训练模型...")
    X, y = create_features(rolls, lookback=10)  # 使用10个历史点数
    print(f"特征矩阵形状: {X.shape}")

    model = train_model(X, y)

    print("\n🚀 模型训练完成！测试预测接口...")
    last_10 = rolls[-10:].tolist()  # 使用最后10局测试
    game_id = len(rolls)
    pred, conf, proba = predict_next_roll(last_10, model, game_id)

    print(f"模型预测下一次点数: {pred}")
    print(f"置信度: {conf:.3f}")
    print(f"各点数概率: {proba.round(3)}")
    
    # 基于规则的预测
    rule_pred, rule_conf, rule_proba = predict_next_roll_rule_based(last_10, game_id)
    print(f"\n规则预测下一次点数: {rule_pred}")
    print(f"置信度: {rule_conf:.3f}")
    print(f"各点数概率: {np.array(rule_proba).round(3)}")

    print("\n🎉 所有分析完成！生成了4张图表：")
    print("  - frequency_distribution.png")
    print("  - autocorrelation.png")
    print("  - transition_matrix.png")
    print("  - weekly_pattern.png")
    print("  - feature_importance.png")


if __name__ == '__main__':
    main()