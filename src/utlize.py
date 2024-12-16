import os
import shutil
import random


def select_one_per_label(source_dir, target_dir):
    """
    扫描源目录，从每个 label 中选择一个文件，复制到目标目录，命名保持不变。

    参数:
        source_dir: 源目录，包含所有图片文件
        target_dir: 目标目录，用于存放选中的图片文件
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    label_dict = {}

    # 遍历所有文件，按 label 进行分类
    for file in os.listdir(source_dir):
        if file.endswith(".png"):  # 确保是图片文件
            label = file.split("-")[0]  # 获取第一个数字作为 label
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(file)

    # 每个 label 随机选择一个文件
    for label, files in label_dict.items():
        chosen_file = random.choice(files)  # 随机选择一个文件
        source_file = os.path.join(source_dir, chosen_file)
        target_file = os.path.join(target_dir, chosen_file)
        shutil.copy(source_file, target_file)  # 复制文件
        print(f"Label {label}: 选择 {chosen_file} -> {target_file}")


if __name__ == "__main__":
    source_directory = r".\data\pre_processed\mel_test"  # 源目录
    target_directory = r".\data\pre_processed\mel_test50"  # 目标目录

    select_one_per_label(source_directory, target_directory)
    print("文件选择与复制完成！")
