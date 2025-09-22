# -*- coding: utf-8 -*-
"""
骰子点数策略分析器
分析特定策略的胜率情况
"""

import numpy as np
from collections import Counter


def load_data(file_path):
    """加载文本数据，每行或连续字符为1-6的数字"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    rolls = [int(char) for char in data if char.isdigit()]
    return np.array(rolls)


def analyze_strategy(rolls):
    """
    分析策略胜率:
    - 如果出现大(4-6)，下一次压小(1-3)
    - 如果出现小(1-3)，下一次压大(4-6)
    - 如果出现单(1,3,5)，下一次压双(2,4,6)
    - 如果出现双(2,4,6)，下一次压单(1,3,5)
    """
    print("开始分析策略胜率...")
    print(f"数据总长度: {len(rolls)}")
    
    # 策略1: 大小交替策略
    big_small_wins = 0
    big_small_total = 0
    
    # 策略2: 单双交替策略
    odd_even_wins = 0
    odd_even_total = 0
    
    # 统计各个策略的使用情况
    big_count = 0  # 出现大点数的次数
    small_count = 0  # 出现小点数的次数
    odd_count = 0   # 出现单数的次数
    even_count = 0  # 出现双数的次数
    
    for i in range(len(rolls) - 1):
        current = rolls[i]
        next_roll = rolls[i + 1]
        
        # 大小策略
        if current >= 4:  # 当前是大
            big_count += 1
            big_small_total += 1
            if next_roll <= 3:  # 下一次是小，策略正确
                big_small_wins += 1
        elif current <= 3:  # 当前是小
            small_count += 1
            big_small_total += 1
            if next_roll >= 4:  # 下一次是大，策略正确
                big_small_wins += 1
                
        # 单双策略
        if current % 2 == 1:  # 当前是单数
            odd_count += 1
            odd_even_total += 1
            if next_roll % 2 == 0:  # 下一次是双数，策略正确
                odd_even_wins += 1
        elif current % 2 == 0:  # 当前是双数
            even_count += 1
            odd_even_total += 1
            if next_roll % 2 == 1:  # 下一次是单数，策略正确
                odd_even_wins += 1
    
    # 计算胜率
    big_small_win_rate = big_small_wins / big_small_total if big_small_total > 0 else 0
    odd_even_win_rate = odd_even_wins / odd_even_total if odd_even_total > 0 else 0
    
    print("\n=== 大小交替策略分析 ===")
    print(f"总预测次数: {big_small_total}")
    print(f"正确预测次数: {big_small_wins}")
    print(f"胜率: {big_small_win_rate:.3f} ({big_small_win_rate*100:.1f}%)")
    print(f"出现大点数的次数: {big_count}")
    print(f"出现小点数的次数: {small_count}")
    
    print("\n=== 单双交替策略分析 ===")
    print(f"总预测次数: {odd_even_total}")
    print(f"正确预测次数: {odd_even_wins}")
    print(f"胜率: {odd_even_win_rate:.3f} ({odd_even_win_rate*100:.1f}%)")
    print(f"出现单数的次数: {odd_count}")
    print(f"出现双数的次数: {even_count}")
    
    # 综合策略 (大小+单双)
    print("\n=== 综合策略分析 ===")
    total_strategies = big_small_total + odd_even_total
    total_wins = big_small_wins + odd_even_wins
    combined_win_rate = total_wins / total_strategies if total_strategies > 0 else 0
    print(f"总预测次数: {total_strategies}")
    print(f"正确预测次数: {total_wins}")
    print(f"胜率: {combined_win_rate:.3f} ({combined_win_rate*100:.1f}%)")
    
    # 随机猜测基准
    print("\n=== 基准对比 ===")
    print("随机猜测胜率: 0.500 (50.0%)")
    print(f"大小策略相对提升: {big_small_win_rate - 0.5:.3f}")
    print(f"单双策略相对提升: {odd_even_win_rate - 0.5:.3f}")
    print(f"综合策略相对提升: {combined_win_rate - 0.5:.3f}")


def analyze_advanced_strategy(rolls):
    """
    分析更复杂的策略:
    - 连续出现相同类型后的反转策略
    """
    print("\n=== 连续模式策略分析 ===")
    
    # 连续大/小后的反转
    consecutive_big_small_wins = 0
    consecutive_big_small_total = 0
    
    # 连续单/双后的反转
    consecutive_odd_even_wins = 0
    consecutive_odd_even_total = 0
    
    # 检查连续模式
    for i in range(1, len(rolls) - 1):
        prev = rolls[i-1]
        current = rolls[i]
        next_roll = rolls[i+1]
        
        # 检查是否是连续的大/小
        if (prev >= 4 and current >= 4) or (prev <= 3 and current <= 3):
            consecutive_big_small_total += 1
            # 预期下一次会反转
            if (current >= 4 and next_roll <= 3) or (current <= 3 and next_roll >= 4):
                consecutive_big_small_wins += 1
                
        # 检查是否是连续的单/双
        if (prev % 2 == current % 2):
            consecutive_odd_even_total += 1
            # 预期下一次会反转
            if (current % 2 != next_roll % 2):
                consecutive_odd_even_wins += 1
                
    # 计算胜率
    cons_big_small_rate = consecutive_big_small_wins / consecutive_big_small_total if consecutive_big_small_total > 0 else 0
    cons_odd_even_rate = consecutive_odd_even_wins / consecutive_odd_even_total if consecutive_odd_even_total > 0 else 0
    
    print(f"连续大小模式预测次数: {consecutive_big_small_total}")
    print(f"连续大小模式正确次数: {consecutive_big_small_wins}")
    print(f"连续大小模式胜率: {cons_big_small_rate:.3f} ({cons_big_small_rate*100:.1f}%)")
    
    print(f"连续单双模式预测次数: {consecutive_odd_even_total}")
    print(f"连续单双模式正确次数: {consecutive_odd_even_wins}")
    print(f"连续单双模式胜率: {cons_odd_even_rate:.3f} ({cons_odd_even_rate*100:.1f}%)")


def analyze_weekly_strategy(rolls, games_per_week=14000):
    """
    按周期分析策略胜率
    """
    print("\n=== 按周期分析策略胜率 ===")
    
    # 分析换骰后前100局的表现
    first_100_wins = 0
    first_100_total = 0
    
    # 分析其他时间的表现
    other_wins = 0
    other_total = 0
    
    for i in range(len(rolls) - 1):
        current = rolls[i]
        next_roll = rolls[i + 1]
        game_id = i + 1
        
        # 计算是本周第几局
        week_game_index = game_id % games_per_week
        
        # 大小策略
        if current >= 4 and next_roll <= 3:  # 大后压小成功
            if week_game_index <= 100:
                first_100_wins += 1
                first_100_total += 1
            else:
                other_wins += 1
                other_total += 1
        elif current <= 3 and next_roll >= 4:  # 小后压大成功
            if week_game_index <= 100:
                first_100_wins += 1
                first_100_total += 1
            else:
                other_wins += 1
                other_total += 1
        elif (current >= 4 and next_roll >= 4) or (current <= 3 and next_roll <= 3):
            # 大小未反转的情况
            if week_game_index <= 100:
                first_100_total += 1
            else:
                other_total += 1
    
    first_100_rate = first_100_wins / first_100_total if first_100_total > 0 else 0
    other_rate = other_wins / other_total if other_total > 0 else 0
    
    print(f"换骰后前100局预测次数: {first_100_total}")
    print(f"换骰后前100局正确次数: {first_100_wins}")
    print(f"换骰后前100局胜率: {first_100_rate:.3f} ({first_100_rate*100:.1f}%)")
    
    print(f"其他时间预测次数: {other_total}")
    print(f"其他时间正确次数: {other_wins}")
    print(f"其他时间胜率: {other_rate:.3f} ({other_rate*100:.1f}%)")


def analyze_transition_probabilities(rolls):
    """
    分析各点数的转移概率
    """
    print("\n=== 各点数转移概率分析 ===")
    
    # 创建转移计数矩阵
    transition_count = np.zeros((6, 6), dtype=int)  # 从点数i到点数j的转移次数
    
    # 统计转移次数
    for i in range(len(rolls) - 1):
        current = rolls[i] - 1  # 转换为0-5索引
        next_roll = rolls[i + 1] - 1  # 转换为0-5索引
        transition_count[current, next_roll] += 1
    
    # 计算转移概率矩阵
    transition_prob = np.zeros((6, 6))
    for i in range(6):
        row_sum = np.sum(transition_count[i])
        if row_sum > 0:
            transition_prob[i] = transition_count[i] / row_sum
    
    # 打印转移概率矩阵
    print("转移概率矩阵 (行=当前点数, 列=下一点数):")
    print("       1\t  2\t  3\t  4\t  5\t  6")
    for i in range(6):
        print(f"  {i+1}  ", end="")
        for j in range(6):
            print(f"{transition_prob[i, j]:.3f}\t", end="")
        print()
    
    # 分析特定策略下的转移情况
    print("\n=== 特定策略下的转移情况 ===")
    
    # 大小策略下的转移
    big_to_small = 0  # 大点数(4-6)转移到小点数(1-3)的次数
    big_to_big = 0    # 大点数转移到大点数的次数
    small_to_big = 0  # 小点数转移到大点数的次数
    small_to_small = 0 # 小点数转移到小点数的次数
    
    big_total = 0  # 大点数出现的总次数
    small_total = 0  # 小点数出现的总次数
    
    for i in range(len(rolls) - 1):
        current = rolls[i]
        next_roll = rolls[i + 1]
        
        if current >= 4:  # 当前是大点数
            big_total += 1
            if next_roll <= 3:  # 转移到小点数
                big_to_small += 1
            else:  # 转移到大点数
                big_to_big += 1
        else:  # 当前是小点数
            small_total += 1
            if next_roll >= 4:  # 转移到大点数
                small_to_big += 1
            else:  # 转移到小点数
                small_to_small += 1
    
    print("大小策略转移概率:")
    print(f"大点数后转移到小点数的概率: {big_to_small / big_total if big_total > 0 else 0:.3f}")
    print(f"大点数后继续保持大点数的概率: {big_to_big / big_total if big_total > 0 else 0:.3f}")
    print(f"小点数后转移到大点数的概率: {small_to_big / small_total if small_total > 0 else 0:.3f}")
    print(f"小点数后继续保持小点数的概率: {small_to_small / small_total if small_total > 0 else 0:.3f}")
    
    # 单双策略下的转移
    odd_to_even = 0   # 单数转移到双数的次数
    odd_to_odd = 0    # 单数转移到单数的次数
    even_to_odd = 0   # 双数转移到单数的次数
    even_to_even = 0  # 双数转移到双数的次数
    
    odd_total = 0     # 单数出现的总次数
    even_total = 0    # 双数出现的总次数
    
    for i in range(len(rolls) - 1):
        current = rolls[i]
        next_roll = rolls[i + 1]
        
        if current % 2 == 1:  # 当前是单数
            odd_total += 1
            if next_roll % 2 == 0:  # 转移到双数
                odd_to_even += 1
            else:  # 转移到单数
                odd_to_odd += 1
        else:  # 当前是双数
            even_total += 1
            if next_roll % 2 == 1:  # 转移到单数
                even_to_odd += 1
            else:  # 转移到双数
                even_to_even += 1
    
    print("\n单双策略转移概率:")
    print(f"单数后转移到双数的概率: {odd_to_even / odd_total if odd_total > 0 else 0:.3f}")
    print(f"单数后继续保持单数的概率: {odd_to_odd / odd_total if odd_total > 0 else 0:.3f}")
    print(f"双数后转移到单数的概率: {even_to_odd / even_total if even_total > 0 else 0:.3f}")
    print(f"双数后继续保持双数的概率: {even_to_even / even_total if even_total > 0 else 0:.3f}")


def main():
    file_path = 'bg_history_result.txt'
    
    print("🎲 开始加载数据...")
    rolls = load_data(file_path)
    print(f"✅ 数据加载完成，共 {len(rolls):,} 条记录\n")
    
    # 基本策略分析
    analyze_strategy(rolls)
    
    # 高级策略分析
    analyze_advanced_strategy(rolls)
    
    # 按周期分析策略
    analyze_weekly_strategy(rolls)
    
    # 转移概率分析
    analyze_transition_probabilities(rolls)


if __name__ == '__main__':
    main()