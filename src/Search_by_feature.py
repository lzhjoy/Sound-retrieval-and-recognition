'''效果十分差！差！差！'''


import os
import numpy as np
from fastdtw import fastdtw  # 使用 fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from MFCC_pure import (
    process_wav_file,
    stft_custom,
    mel_custom,
    mfcc_custom,
)


def extract_feature(file_path, sr=None, duration=5, feature_type="MFCC"):
    """
    提取指定的特征图：STFT, Mel, MFCC
    参数:
        file_path: 音频文件路径
        sr: 采样率
        duration: 截取时长
        feature_type: 特征提取类型
    返回:
        提取的特征
    """
    # 预处理音频信号 (归一化 + 预加重)
    data, sample_rate = process_wav_file(file_path, sr=sr, duration=duration)

    # 计算特征
    if feature_type == "STFT":
        feature = stft_custom(data)  # STFT 特征
    elif feature_type == "Mel":
        stft_matrix = stft_custom(data)  # 先计算 STFT
        feature = mel_custom(stft_matrix, sr=sample_rate, n_fft=1024)  # Mel 特征
    elif feature_type == "MFCC":
        stft_matrix = stft_custom(data)  # 先计算 STFT
        mel_spectrogram = mel_custom(
            stft_matrix, sr=sample_rate, n_fft=1024
        )  # Mel 特征
        feature = mfcc_custom(mel_spectrogram)  # MFCC 特征
    else:
        raise ValueError("Invalid feature_type. Choose from 'STFT', 'Mel', or 'MFCC'.")
    return feature


def dtw_distance(feature1, feature2):
    """
    使用 fastdtw 计算两个特征之间的DTW距离
    参数:
        feature1: 特征1
        feature2: 特征2
    返回:
        DTW 距离
    """
    distance, _ = fastdtw(feature1.T, feature2.T, dist=euclidean)
    return distance


def process_data(train_dir, test_dir, feature_type="MFCC"):
    """
    处理数据，计算DTW距离并获取Top10/Top20
    参数:
        train_dir: 训练集目录
        test_dir: 测试集目录
        feature_type: 特征类型 (STFT, Mel, MFCC)
    """
    train_files = [f for f in os.listdir(train_dir) if f.endswith(".wav")]
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]

    # 读取训练集特征
    print("正在提取训练集特征...")
    train_features = []
    train_info = []  # 存储文件名和标签
    for file in tqdm(train_files, desc="训练集"):
        path = os.path.join(train_dir, file)
        feature = extract_feature(path, feature_type=feature_type)
        label = file.split("-")[-1].split(".")[0]
        train_features.append((feature, label))
        train_info.append((file, label))  # 保存文件名和标签

    # 对测试集计算DTW
    print("正在处理测试集并计算DTW距离...")
    top10_accuracy, top20_accuracy = [], []
    for test_file in tqdm(test_files, desc="测试集"):
        test_path = os.path.join(test_dir, test_file)
        test_feature = extract_feature(test_path, feature_type=feature_type)
        test_label = test_file.split("-")[-1].split(".")[0]

        # 计算DTW距离，并添加进度条
        distances = []
        for idx, (train_feature, train_label) in enumerate(
            tqdm(train_features, desc=f"DTW计算: {test_file}", leave=False)
        ):
            dist = dtw_distance(test_feature, train_feature)
            distances.append((dist, train_info[idx][0], train_label))

        # 获取Top10和Top20
        distances.sort(key=lambda x: x[0])
        top10 = distances[:10]
        top20 = distances[:20]

        # 提取Top10和Top20的标签
        top10_labels = [label for _, _, label in top10]
        top20_labels = [label for _, _, label in top20]

        # 计算精度
        top10_accuracy.append(top10_labels.count(test_label) / 10)
        top20_accuracy.append(top20_labels.count(test_label) / 20)

        print(
            f"\nFile: {test_file}\n"
            f"Top10: {[name for _, name, _ in top10]}\n"
            f"Top20: {[name for _, name, _ in top20]}\n"
            f"Top10 Accuracy: {top10_labels.count(test_label)/10:.2f}, "
            f"Top20 Accuracy: {top20_labels.count(test_label)/20:.2f}\n"
        )

    # 平均精度
    avg_top10 = np.mean(top10_accuracy)
    avg_top20 = np.mean(top20_accuracy)
    print(
        f"Overall Top10 Accuracy: {avg_top10:.2f}, Overall Top20 Accuracy: {avg_top20:.2f}"
    )


if __name__ == "__main__":
    train_directory = r"./data/raw/train"  # 训练集目录
    test_directory = r"./data/raw/test"  # 测试集目录
    feature_type = "MFCC"  # 可选：'STFT', 'Mel', 'MFCC'

    process_data(train_directory, test_directory, feature_type)
