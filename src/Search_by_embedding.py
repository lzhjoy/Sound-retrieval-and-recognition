import os
import torch
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from PIL import Image


def load_mel_from_png(png_path):
    """
    从PNG图像中加载Mel频谱图，并转换为灰度数值矩阵
    """
    img = Image.open(png_path).convert("L")
    return np.array(img)


def extract_label(file_name):
    """
    从文件名中提取label（第一个数字）
    """
    return int(file_name.split("-")[0])


def load_pretrained_model():
    """
    加载 Hugging Face 上的预训练视觉模型和特征提取器
    """
    model_name = "google/vit-base-patch16-224-in21k"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, feature_extractor


def get_embedding(model, feature_extractor, image):
    """
    将Mel频谱图转换为模型的嵌入表示
    """
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


def top_k_and_accuracy(
    mel_test_dir, mel_pure_dir, model, feature_extractor, top_k1=10, top_k2=20
):
    """
    对mel_test目录下的频谱图与mel_pure进行嵌入比较，找出Top10和Top20，并实时计算精度
    """
    # 获取所有测试和候选频谱图文件
    test_files = sorted(os.listdir(mel_test_dir))
    pure_files = sorted(os.listdir(mel_pure_dir))

    if not test_files or not pure_files:
        print("测试目录或候选数据库目录为空，请检查路径。")
        return

    print("\n开始检索和实时精度计算任务...")

    # 加载候选数据库的Mel频谱图
    pure_embeddings = []
    pure_labels = []
    for pure_file in tqdm(pure_files, desc="加载候选数据库"):
        pure_path = os.path.join(mel_pure_dir, pure_file)
        pure_mel = load_mel_from_png(pure_path)
        embedding = get_embedding(model, feature_extractor, pure_mel)
        pure_embeddings.append(embedding)
        pure_labels.append(extract_label(pure_file))

    pure_embeddings = np.array(pure_embeddings)

    # 遍历测试集，计算相似度
    top10_correct, top20_correct = 0, 0
    for idx, test_file in enumerate(tqdm(test_files, desc="处理测试样本")):
        test_label = extract_label(test_file)
        test_path = os.path.join(mel_test_dir, test_file)
        test_mel = load_mel_from_png(test_path)
        test_embedding = get_embedding(model, feature_extractor, test_mel)

        # 计算余弦相似度
        similarities = cosine_similarity([test_embedding], pure_embeddings)[0]
        sorted_indices = similarities.argsort()[::-1]  # 降序排序

        # 提取Top10和Top20
        top10_labels = [pure_labels[i] for i in sorted_indices[:top_k1]]
        top20_labels = [pure_labels[i] for i in sorted_indices[:top_k2]]

        # 计算精度
        top10_correct += top10_labels.count(test_label)
        top20_correct += top20_labels.count(test_label)

        current_top10_accuracy = top10_correct / ((idx + 1) * top_k1)
        current_top20_accuracy = top20_correct / ((idx + 1) * top_k2)

        print(f"\n当前进度: {idx + 1}/{len(test_files)}")
        print(f"当前 Top 10 精度: {current_top10_accuracy * 100:.2f}%")
        print(f"当前 Top 20 精度: {current_top20_accuracy * 100:.2f}%")

    # 最终精度
    print("\n最终精度:")
    print(f"Top 10 精度: {top10_correct / (len(test_files) * top_k1) * 100:.2f}%")
    print(f"Top 20 精度: {top20_correct / (len(test_files) * top_k2) * 100:.2f}%")


# 主程序
if __name__ == "__main__":
    mel_test_directory = r"./data/pre_processed/mel_test"  # 测试目录
    mel_pure_directory = r"./data/pre_processed/mel_pure"  # 候选数据库目录

    model, feature_extractor = load_pretrained_model()
    top_k_and_accuracy(
        mel_test_directory,
        mel_pure_directory,
        model,
        feature_extractor,
        top_k1=10,
        top_k2=20,
    )
