import os
import matplotlib.pyplot as plt
from video_processor import DiceVideoProcessor
from pathlib import Path  # 增加导入


def visualize_results(y_true, y_pred):
    """可视化预测结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, 'b-', label='实际值')
    plt.plot(y_pred, 'r--', label='预测值')
    plt.xlabel('样本')
    plt.ylabel('骰子点数')
    plt.title('骰子点数预测结果')
    plt.legend()
    plt.grid(True)

    plt.savefig('prediction_results.png')
    plt.show()


def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    # 1. 处理视频
    # video_path = "C:\\Users\\fred\\Desktop\\Fred\\bg-game\\video_1741004387.flv"

    # 修改后代码
    folder = Path("C:\\Users\\fred\\Desktop\\Fred\\bg-game")
    processor = DiceVideoProcessor()
    print("正在处理视频...")
    roi = [514, 134, 224, 224]
    # 获取所有flv文件（包括子目录）
    flv_files = [f for f in folder.glob('**/*.flv') if f.is_file()]
    # 仅当前目录（不含子目录）：
    # flv_files = [f for f in folder.glob('*.flv') if f.is_file()]
    for f in flv_files:
        try:
            print(f"正在处理: {f.name}")
            processor.process_video(str(f), roi=roi)  # 转换为字符串路径
        except Exception as e:
            print(f"处理文件 {f} 失败: {str(e)}")
            continue  # 记录日志后继续处理下一个文件

if __name__ == "__main__":
    main()
