import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import librosa
import librosa.display
from tqdm import tqdm

# 动态配置中文字体
import matplotlib

matplotlib.rcParams["font.family"] = "SimHei"  # 设置中文字体
matplotlib.rcParams["axes.unicode_minus"] = False  # 正确显示负号


def pre_emphasis(signal, alpha=0.97):
    """
    预加重处理：高频信号增强
    参数:
        signal: 输入音频信号
        alpha: 预加重系数 (默认0.97)
    返回:
        预加重后的信号
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def normalize_signal(signal):
    """
    对信号进行归一化处理
    参数:
        signal: 输入音频信号
    返回:
        归一化后的信号
    """
    return signal / np.max(np.abs(signal))


def batch_stft_processing(
    directory_path,
    duration=5,  # 音频截取长度（秒）
    window_size=1024,  # 窗口大小
    hop_size=512,  # 步长
    n_mels=128,  # Mel滤波器数量
    fmin=0,  # 最小频率
    fmax=None,  # 最大频率
    n_mfcc=13,  # MFCC通道数量
    save_plots_stft=False,
    save_plots_mel=False,
    save_plots_mfcc=False,
    visualize_mel_filters=False,
    output_dir_stft=None,
    output_dir_mel=None,
    output_dir_mfcc=None,
):
    # 获取目录下的所有wav文件
    wav_files = glob.glob(os.path.join(directory_path, "*.wav"))
    if not wav_files:
        print(f"在目录 {directory_path} 中未找到任何 .wav 文件。")
        return

    for file_path in tqdm(wav_files):
        print(f"处理文件: {file_path}")

        # 读取音频文件并截取指定时长
        try:
            data, sample_rate = librosa.load(file_path, sr=None, duration=duration)
        except Exception as e:
            print(f"无法读取文件 {file_path}，错误: {e}")
            continue

        file_name = os.path.basename(file_path)

        # === 归一化处理 ===
        data = normalize_signal(data)

        # === 预加重处理 ===
        data = pre_emphasis(data)

        # === 1. 计算 STFT ===
        stft_matrix = librosa.stft(data, n_fft=window_size, hop_length=hop_size)
        stft_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)

        # 绘制STFT幅度谱
        title = f"STFT 幅度谱 - {file_name}"
        if save_plots_stft:
            if output_dir_stft is None:
                output_dir_stft = os.path.join(directory_path, "stft_plots")
                os.makedirs(output_dir_stft, exist_ok=True)
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                stft_db, sr=sample_rate, hop_length=hop_size, x_axis="time", y_axis="hz"
            )
            plt.title(title)
            plt.colorbar(label="幅度 (dB)")
            plot_path = os.path.join(output_dir_stft, f"{file_name}_stft.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"已保存STFT图表到: {plot_path}")
        else:
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                stft_db, sr=sample_rate, hop_length=hop_size, x_axis="time", y_axis="hz"
            )
            plt.title(title)
            plt.colorbar(label="幅度 (dB)")
            plt.show()

        # === 2. 生成 Mel 滤波器组 ===
        if fmax is None:
            fmax = sample_rate / 2

        mel_filter = librosa.filters.mel(
            sr=sample_rate, n_fft=window_size, n_mels=n_mels, fmin=fmin, fmax=fmax
        )

        # 可视化Mel滤波器
        if visualize_mel_filters:
            plt.figure(figsize=(12, 6))
            for i in range(n_mels):
                plt.plot(np.linspace(0, fmax, mel_filter.shape[1]), mel_filter[i])
            plt.title("Mel 滤波器组频率响应")
            plt.xlabel("频率 (Hz)")
            plt.ylabel("滤波器增益")
            plt.grid(True)
            plt.show()

        # === 3. 计算 Mel 频谱图 ===
        mel_spectrogram = librosa.feature.melspectrogram(
            y=data,
            sr=sample_rate,
            n_fft=window_size,
            hop_length=hop_size,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # 绘制Mel频谱图
        mel_title = f"Mel 频谱图 - {file_name}"
        if save_plots_mel:
            if output_dir_mel is None:
                output_dir_mel = os.path.join(directory_path, "mel_plots")
                os.makedirs(output_dir_mel, exist_ok=True)
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                mel_db, sr=sample_rate, hop_length=hop_size, x_axis="time", y_axis="mel"
            )
            plt.title(mel_title)
            plt.colorbar(label="幅度 (dB)")
            plot_path = os.path.join(output_dir_mel, f"{file_name}_mel.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"已保存Mel图表到: {plot_path}")
        else:
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                mel_db, sr=sample_rate, hop_length=hop_size, x_axis="time", y_axis="mel"
            )
            plt.title(mel_title)
            plt.colorbar(label="幅度 (dB)")
            plt.show()

        # === 4. 计算 MFCC 特征 ===
        mfcc = librosa.feature.mfcc(S=mel_spectrogram, sr=sample_rate, n_mfcc=n_mfcc)

        # 均值归一化
        mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (
            np.std(mfcc, axis=1, keepdims=True) + 1e-10
        )

        # 绘制MFCC特征图
        mfcc_title = f"MFCC 特征图 - {file_name}"
        if save_plots_mfcc:
            if output_dir_mfcc is None:
                output_dir_mfcc = os.path.join(directory_path, "mfcc_plots")
                os.makedirs(output_dir_mfcc, exist_ok=True)
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                mfcc_normalized, sr=sample_rate, hop_length=hop_size, x_axis="time"
            )
            plt.title(mfcc_title)
            plt.colorbar(label="MFCC系数值")
            plot_path = os.path.join(output_dir_mfcc, f"{file_name}_mfcc.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"已保存MFCC图表到: {plot_path}")
        else:
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                mfcc_normalized, sr=sample_rate, hop_length=hop_size, x_axis="time"
            )
            plt.title(mfcc_title)
            plt.colorbar(label="MFCC系数值")
            plt.show()


if __name__ == "__main__":
    wav_directory = r"./data/raw/train"

    batch_stft_processing(
        directory_path=wav_directory,
        duration=5,
        window_size=1024,
        hop_size=512,
        n_mels=128,
        fmin=0,
        fmax=None,
        n_mfcc=13,
        save_plots_stft=True,
        save_plots_mel=True,
        save_plots_mfcc=True,
        output_dir_mel=r".\data\pre_processed\mel",
        output_dir_stft=r".\data\pre_processed\stft",
        output_dir_mfcc=r".\data\pre_processed\mfcc",
        visualize_mel_filters=False,
    )
