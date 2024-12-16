"""计算复杂度实在是太高了！跑不下来"""

import os
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import numpy as np


def extract_label(filename):
    """
    从文件名中提取标签 (最后一个数字)
    参数:
        filename: 文件名 (例如: "1-137-A-32.wav")
    返回:
        标签 (字符串形式)
    """
    return filename.split("-")[-1].split(".")[0]


def load_wav(file_path, sr=None):
    """
    加载音频文件并归一化
    参数:
        file_path: 文件路径
        sr: 采样率
    返回:
        音频信号数组
    """
    y, _ = librosa.load(file_path, sr=sr)
    return librosa.util.normalize(y)


def dtw_distance(signal1, signal2):
    """
    计算两个音频信号之间的DTW距离
    参数:
        signal1, signal2: 输入的音频信号
    返回:
        DTW距离
    """
    # 确保输入是 1-D 数组
    # signal1 = signal1.flatten().tolist()
    # signal2 = signal2.flatten().tolist()

    signal1 = signal1.reshape(-1, 1)
    signal2 = signal2.reshape(-1, 1)
    print(signal1.shape)
    distance, _ = fastdtw(signal1, signal2, dist=euclidean)
    return distance


def process_data(train_dir, test_dir):
    """
    计算DTW距离并获取Top10/Top20精度
    参数:
        train_dir: 训练集目录
        test_dir: 测试集目录
    """
    # 获取训练集和测试集文件
    train_files = [f for f in os.listdir(train_dir) if f.endswith(".wav")]
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]

    # 加载训练集音频
    print("加载训练集...")
    train_data = []
    for train_file in tqdm(train_files, desc="训练集加载"):
        train_path = os.path.join(train_dir, train_file)
        train_signal = load_wav(train_path)
        train_label = extract_label(train_file)
        train_data.append((train_signal, train_label, train_file))

    # 初始化精度
    top10_accuracies = []
    top20_accuracies = []

    # 处理测试集
    print("处理测试集...")
    for test_file in tqdm(test_files, desc="测试集处理"):
        test_path = os.path.join(test_dir, test_file)
        test_signal = load_wav(test_path)
        test_label = extract_label(test_file)

        # 计算DTW距离
        distances = []
        for train_signal, train_label, train_file in tqdm(
            train_data, desc=f"DTW计算: {test_file}", leave=False
        ):
            dist = dtw_distance(test_signal, train_signal)
            distances.append((dist, train_label, train_file))

        # 根据距离排序
        distances.sort(key=lambda x: x[0])

        # 提取Top10和Top20
        top10 = distances[:10]
        top20 = distances[:20]

        top10_labels = [label for _, label, _ in top10]
        top20_labels = [label for _, label, _ in top20]

        # 计算精度
        top10_accuracy = top10_labels.count(test_label) / 10
        top20_accuracy = top20_labels.count(test_label) / 20

        top10_accuracies.append(top10_accuracy)
        top20_accuracies.append(top20_accuracy)

        # 输出当前测试文件结果
        print(
            f"\n文件: {test_file}\n"
            f"Top10: {[file for _, _, file in top10]}\n"
            f"Top20: {[file for _, _, file in top20]}\n"
            f"Top10 精度: {top10_accuracy:.2f}, Top20 精度: {top20_accuracy:.2f}\n"
        )

    # 输出总体精度
    avg_top10 = np.mean(top10_accuracies)
    avg_top20 = np.mean(top20_accuracies)
    print(f"总体 Top10 精度: {avg_top10:.2f}")
    print(f"总体 Top20 精度: {avg_top20:.2f}")


if __name__ == "__main__":
    train_directory = "./data/raw/train"  # 训练集路径
    test_directory = "./data/raw/test"  # 测试集路径
    process_data(train_directory, test_directory)
