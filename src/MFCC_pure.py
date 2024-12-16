import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import librosa
import librosa.display
from scipy.fftpack import dct
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


def process_wav_file(file_path, duration=5, sr=None, pre_emphasis_coeff=0.97):
    """
    读取和预处理音频文件
    参数:
        file_path: 音频文件路径
        duration: 截取时长（秒）
        sr: 采样率
        pre_emphasis_coeff: 预加重系数
    返回:
        预处理后的音频信号和采样率
    """
    data, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
    data = normalize_signal(data)
    data = pre_emphasis(data, alpha=pre_emphasis_coeff)
    return data, sample_rate


def stft_custom(data, n_fft=1024, hop_length=512):
    """
    计算 STFT
    参数:
        data: 输入音频信号
        n_fft: FFT 窗口大小
        hop_length: 窗口步长
    返回:
        STFT 幅度谱 (dB)
    """
    stft_matrix = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)


def mel_custom(stft_matrix, sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """
    计算 Mel 频谱图
    参数:
        stft_matrix: STFT 幅度谱
        sr: 采样率
        n_fft: FFT 窗口大小
        n_mels: Mel 滤波器数量
        fmin: 最小频率
        fmax: 最大频率
    返回:
        Mel 频谱图 (dB)
    """
    mel_filter = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    mel_spectrogram = np.dot(mel_filter, np.abs(stft_matrix) ** 2)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def mfcc_custom(mel_spectrogram, n_mfcc=13):
    """
    提取 MFCC 特征
    参数:
        mel_spectrogram: Mel 频谱图 (功率谱)
        n_mfcc: MFCC 通道数量
    返回:
        MFCC 特征矩阵
    """
    mfcc = dct(mel_spectrogram, type=2, axis=0, norm="ortho")[:n_mfcc]
    mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (
        np.std(mfcc, axis=1, keepdims=True) + 1e-10
    )
    return mfcc_normalized


def save_stft_plot_clean(stft_db, output_path):
    """
    绘制并保存无标签、无标题的 STFT 频谱图
    参数:
        stft_db: STFT 的幅度谱 (dB)
        output_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    plt.axis("off")  # 去除坐标轴
    plt.imshow(stft_db, aspect="auto", origin="lower", cmap="plasma")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_mel_plot_clean(mel_db, output_path):
    """
    绘制并保存无标签、无标题的 Mel 频谱图
    参数:
        mel_db: Mel 频谱图 (dB)
        output_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    plt.axis("off")  # 去除坐标轴
    plt.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_mfcc_plot_clean(mfcc_features, output_path):
    """
    绘制并保存无标签、无标题的 MFCC 特征图
    参数:
        mfcc_features: MFCC 特征矩阵
        output_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    plt.axis("off")  # 去除坐标轴
    plt.imshow(mfcc_features, aspect="auto", origin="lower", cmap="viridis")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def batch_stft_processing(
    directory_path,
    duration=5,
    window_size=1024,
    hop_size=512,
    n_mels=128,
    fmin=0,
    fmax=None,
    n_mfcc=13,
    save_plots_stft=False,
    save_plots_mel=False,
    save_plots_mfcc=False,
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

        # 提取 label 和重命名文件
        file_name = os.path.basename(file_path)
        parts = file_name.split("-")
        label = parts[-1].split(".")[0]  # 提取最后的数字部分作为 label
        new_name = f"{label}-{'-'.join(parts[:-1])}"  # 组合新的文件名

        # 预处理音频
        data, sample_rate = process_wav_file(file_path, duration=duration)

        # 计算 STFT
        stft_db = stft_custom(data, n_fft=window_size, hop_length=hop_size)

        # 绘制无标注 STFT 图
        if save_plots_stft:
            if output_dir_stft is None:
                output_dir_stft = os.path.join(directory_path, "stft_plots")
                os.makedirs(output_dir_stft, exist_ok=True)
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                stft_db, sr=sample_rate, hop_length=hop_size, x_axis=None, y_axis=None
            )
            plt.axis("off")  # 去掉坐标轴
            plt.savefig(
                os.path.join(output_dir_stft, f"{new_name}_stft.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

        # 计算 Mel 频谱图
        mel_db = mel_custom(
            stft_matrix=np.abs(
                librosa.stft(data, n_fft=window_size, hop_length=hop_size)
            ),
            sr=sample_rate,
            n_fft=window_size,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        # 绘制无标注 Mel 频谱图
        if save_plots_mel:
            if output_dir_mel is None:
                output_dir_mel = os.path.join(directory_path, "mel_plots")
                os.makedirs(output_dir_mel, exist_ok=True)
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                mel_db, sr=sample_rate, hop_length=hop_size, x_axis=None, y_axis=None
            )
            plt.axis("off")  # 去掉坐标轴
            plt.savefig(
                os.path.join(output_dir_mel, f"{new_name}_mel.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

        # 计算 MFCC
        mfcc_features = mfcc_custom(mel_db, n_mfcc=n_mfcc)

        # 绘制无标注 MFCC 特征图
        if save_plots_mfcc:
            if output_dir_mfcc is None:
                output_dir_mfcc = os.path.join(directory_path, "mfcc_plots")
                os.makedirs(output_dir_mfcc, exist_ok=True)
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(
                mfcc_features, sr=sample_rate, hop_length=hop_size, x_axis=None
            )
            plt.axis("off")  # 去掉坐标轴
            plt.savefig(
                os.path.join(output_dir_mfcc, f"{new_name}_mfcc.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


if __name__ == "__main__":
    batch_stft_processing(
        directory_path="./data/raw/test",
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
        output_dir_stft="./data/pre_processed/stft_test",
        output_dir_mel="./data/pre_processed/mel_test",
        output_dir_mfcc="./data/pre_processed/mfcc_test",
    )
