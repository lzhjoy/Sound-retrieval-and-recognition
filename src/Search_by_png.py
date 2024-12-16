"""效果还算不错，可以跑50个"""

import os
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt  # 导入 matplotlib

import matplotlib

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False


def load_mel_from_png(png_path):
    """
    从PNG图像中加载Mel频谱图，并转换为灰度数值矩阵
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法加载图像: {png_path}")
    img_normalized = img / 255.0
    return img_normalized


def extract_label(file_name):
    """
    从文件名中提取label（第一个数字）
    """
    return int(file_name.split("-")[0])


def compute_dtw_distance(matrix1, matrix2):
    """
    使用DTW计算两个Mel频谱图之间的相似度
    """
    distance, _ = fastdtw(matrix1.T, matrix2.T, dist=euclidean)
    return distance


def dtw_top_k_and_accuracy(mel_test_dir, mel_pure_dir, top_k1=10, top_k2=20):
    """
    对mel_test目录下的频谱图与mel_pure进行DTW比较，找出Top10和Top20，并实时计算精度，并记录到文件
    """
    # 获取所有测试和候选频谱图文件
    test_files = sorted(os.listdir(mel_test_dir))
    pure_files = sorted(os.listdir(mel_pure_dir))

    if not test_files or not pure_files:
        print("测试目录或候选数据库目录为空，请检查路径。")
        return

    # 定义结果保存路径
    result_file = r".\output\accuracy_results.txt"

    # 打开文件用于写入结果
    with open(result_file, "w", encoding="utf-8") as f:
        print("\n开始检索和实时精度计算任务...")
        f.write("开始检索和实时精度计算任务...\n\n")
        f.flush()  # 强制刷新，将缓冲区的数据写入文件
        # 加载候选数据库的Mel频谱图
        pure_mel_dict = {}
        for pure_file in tqdm(pure_files, desc="加载候选数据库"):
            pure_path = os.path.join(mel_pure_dir, pure_file)
            pure_mel_dict[pure_file] = load_mel_from_png(pure_path)

        # 初始化准确率存储
        top10_correct = 0
        top20_correct = 0

        # 遍历测试集，计算DTW相似度
        for idx, test_file in enumerate(tqdm(test_files, desc="处理测试样本")):
            test_label = extract_label(test_file)
            test_path = os.path.join(mel_test_dir, test_file)
            test_mel = load_mel_from_png(test_path)

            # 打印当前正在测试的文件名
            current_file_info = f"\n正在测试文件: {test_file} (标签: {test_label})\n"
            print(current_file_info)
            f.write(current_file_info)
            f.flush()

            # 计算与候选数据库的DTW距离
            distances = []
            for pure_file, pure_mel in tqdm(pure_mel_dict.items()):
                distance = compute_dtw_distance(test_mel, pure_mel)
                distances.append((pure_file, distance))

            # 排序并提取Top10和Top20
            distances.sort(key=lambda x: x[1])
            top10 = distances[:top_k1]
            top20 = distances[:top_k2]

            # 精度计算
            top10_labels = [extract_label(item[0]) for item in top10]
            top20_labels = [extract_label(item[0]) for item in top20]

            current_top10_hits = top10_labels.count(test_label)  # Top10中标签匹配次数
            current_top20_hits = top20_labels.count(test_label)  # Top20中标签匹配次数
            current_top10_accuracy = current_top10_hits / top_k1  # 当前Top10精度
            current_top20_accuracy = current_top20_hits / top_k2  # 当前Top20精度
            top10_correct += top10_labels.count(test_label)
            top20_correct += top20_labels.count(test_label)

            # 实时打印Top10和Top20结果
            top10_info = "\nTop 10 结果:\n" + "\n".join(
                [
                    f"{i+1}. 文件: {file}, 距离: {dist:.2f}"
                    for i, (file, dist) in enumerate(top10)
                ]
            )
            top20_info = "\nTop 20 结果:\n" + "\n".join(
                [
                    f"{i+1}. 文件: {file}, 距离: {dist:.2f}"
                    for i, (file, dist) in enumerate(top20)
                ]
            )

            # 实时打印并写入文件
            progress_info = (
                f"当前进度: {idx + 1}/{len(test_files)}\n"
                f"正在测试文件: {test_file} (标签: {test_label})\n"
                f"当前 Top 10 精度: {current_top10_accuracy * 100:.2f}%\n"
                f"当前 Top 20 精度: {current_top20_accuracy * 100:.2f}%\n"
                f"{top10_info}\n{top20_info}\n"
            )
            print(progress_info)
            f.write(progress_info)
            f.flush()  # 强制刷新，将缓冲区的数据写入文件

        # 最终精度
        final_accuracy = (
            f"\n最终精度:\n"
            f"Top 10 精度: {top10_correct / (len(test_files) * top_k1) * 100:.2f}%\n"
            f"Top 20 精度: {top20_correct / (len(test_files) * top_k2) * 100:.2f}%\n"
        )
        print(final_accuracy)
        f.write(final_accuracy)

    print(f"\n结果已保存到文件: {result_file}")


# 主程序
if __name__ == "__main__":
    mel_test_directory = r".\data\pre_processed\mel_test50"  # 测试目录
    mel_pure_directory = r".\data\pre_processed\mel_pure"  # 候选数据库目录

    dtw_top_k_and_accuracy(mel_test_directory, mel_pure_directory, top_k1=10, top_k2=20)
