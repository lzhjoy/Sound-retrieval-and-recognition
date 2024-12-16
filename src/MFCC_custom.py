import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import glob
from scipy.fftpack import dct
from tqdm import tqdm

import matplotlib

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False


def fft_iterative(x):
    N = len(x)
    X = np.array(x, dtype=np.complex64)
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        if j < bit:
            j += bit
        if i < j:
            X[i], X[j] = X[j], X[i]
    m = 2
    while m <= N:
        theta = -2j * np.pi / m
        wm = np.exp(theta)
        for k in range(0, N, m):
            w = 1
            for j in range(m // 2):
                t = w * X[k + j + m // 2]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + m // 2] = u - t
                w *= wm
        m *= 2
    return X


def next_power_of_two(x):
    return 1 << (x - 1).bit_length()


def pad_to_power_of_two(x):
    N = len(x)
    N_padded = next_power_of_two(N)
    if N_padded == N:
        return x
    else:
        return np.pad(x, (0, N_padded - N), "constant")


def hann_window(N):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)


def plot_stft(
    time_bins, freq_bins, magnitude_db, title, cmap="plasma", vmin=-100, vmax=0
):
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(
        time_bins,
        freq_bins,
        magnitude_db,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.ylabel("频率 (Hz)")
    plt.xlabel("时间 (秒)")
    plt.colorbar(label="幅度 (dB)")
    plt.ylim(0, np.max(freq_bins))
    plt.show()


def plot_mel_spectrogram(
    time_bins, mel_freqs, mel_db, title, cmap="magma", vmin=-100, vmax=0
):
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(
        time_bins, mel_freqs, mel_db, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
    )
    plt.title(title)
    plt.ylabel("Mel 频率通道")
    plt.xlabel("时间 (秒)")
    plt.colorbar(label="幅度 (dB)")
    plt.show()


def plot_mfcc(time_bins, mfcc, title, cmap="viridis"):
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(
        time_bins, np.arange(mfcc.shape[0]), mfcc, shading="gouraud", cmap=cmap
    )
    plt.title(title)
    plt.ylabel("MFCC 通道")
    plt.xlabel("时间 (秒)")
    plt.colorbar(label="MFCC系数值")
    plt.show()


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filter_banks(sr, n_fft, n_mels=128, fmin=0, fmax=None, norm="slaney"):
    """
    模仿Librosa库完全实现Mel滤波器组 (Slaney归一化)
    """
    if fmax is None:
        fmax = sr / 2

    # 1. Mel频率点映射到Hz，并生成频率边界
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # 2. 确定频率轴与频率bin
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    bins = np.searchsorted(fft_freqs, hz_points)

    # 3. 构建滤波器
    filter_bank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]

        # 左斜率
        filter_bank[m - 1, f_left:f_center] = (
            fft_freqs[f_left:f_center] - hz_points[m - 1]
        ) / (hz_points[m] - hz_points[m - 1])
        # 右斜率
        filter_bank[m - 1, f_center:f_right] = (
            hz_points[m + 1] - fft_freqs[f_center:f_right]
        ) / (hz_points[m + 1] - hz_points[m])

    # 4. Slaney归一化：滤波器总和调整
    if norm == "slaney":
        enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
        filter_bank *= enorm[:, np.newaxis]

    return filter_bank, fft_freqs


def process_wav_file(file_path, duration=5, pre_emphasis=0.97):
    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        print(f"无法读取文件 {file_path}，错误: {e}")
        return None, None, None

    if len(data.shape) > 1:
        data = data[:, 0]

    N = int(sample_rate * duration)
    if N > len(data):
        print(
            f"文件 {file_path} 长度不足 {duration} 秒，实际长度: {len(data)/sample_rate:.2f} 秒"
        )
        N = len(data)
    data = data[:N]

    if np.max(np.abs(data)) == 0:
        print(f"文件 {file_path} 的数据全为零，跳过处理。")
        return None, None, None
    data = data / np.max(np.abs(data))

    # 预加重
    data_preemph = np.append(data[0], data[1:] - pre_emphasis * data[:-1])

    return data_preemph, sample_rate, N


def stft_custom(x, sample_rate, window_size=1024, hop_size=512):
    window = hann_window(window_size)
    num_frames = 1 + (len(x) - window_size) // hop_size
    stft_matrix = np.zeros((window_size // 2 + 1, num_frames), dtype=np.float32)
    time_bins = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        frame = x[start:end]
        if len(frame) < window_size:
            frame = np.pad(frame, (0, window_size - len(frame)), "constant")
        frame_win = frame * window
        frame_padded = pad_to_power_of_two(frame_win)
        X = fft_iterative(frame_padded)
        X = X[: window_size // 2 + 1]
        magnitude = np.abs(X) * 2 / window_size
        stft_matrix[:, i] = magnitude
        time_bins[i] = start / sample_rate

    freq_bins = np.linspace(0, sample_rate / 2, window_size // 2 + 1)
    return time_bins, freq_bins, stft_matrix


def mel_custom(sr, n_fft, n_mels=128, fmin=0, fmax=None, norm="slaney"):
    """
    自定义 Mel 滤波器组计算，返回滤波器组和频率轴。
    """
    if fmax is None:
        fmax = sr / 2

    # Mel 频率点映射到 Hz，并生成频率边界
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # 频率轴与频率 bin 对应
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    bins = np.searchsorted(fft_freqs, hz_points)

    # 构建滤波器组
    filter_bank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]

        # 左斜率
        filter_bank[m - 1, f_left:f_center] = (
            fft_freqs[f_left:f_center] - hz_points[m - 1]
        ) / (hz_points[m] - hz_points[m - 1])
        # 右斜率
        filter_bank[m - 1, f_center:f_right] = (
            hz_points[m + 1] - fft_freqs[f_center:f_right]
        ) / (hz_points[m + 1] - hz_points[m])

    # Slaney 归一化：滤波器总和调整
    if norm == "slaney":
        enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
        filter_bank *= enorm[:, np.newaxis]

    return filter_bank, fft_freqs


def mfcc_custom(stft_power, mel_filter, n_mfcc=13):
    """
    自定义 MFCC 特征计算。
    - stft_power: STFT 的功率谱
    - mel_filter: Mel 滤波器组
    - n_mfcc: 返回的 MFCC 通道数
    """
    # 应用 Mel 滤波器
    mel_spectrogram_power = np.dot(mel_filter, stft_power)

    # 进行 DCT 提取 MFCC 特征
    mfcc = dct(mel_spectrogram_power, type=2, axis=0, norm="ortho")[:n_mfcc, :]

    # 均值归一化
    mfcc -= np.mean(mfcc, axis=1, keepdims=True)

    return mfcc


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
    visualize_mel_filters=False,  # 新增参数，用于控制是否可视化mel滤波器
    output_dir_stft=None,
    output_dir_mel=None,
    output_dir_mfcc=None,
):
    wav_files = glob.glob(os.path.join(directory_path, "*.wav"))
    if not wav_files:
        print(f"在目录 {directory_path} 中未找到任何 .wav 文件。")
        return

    for file_path in tqdm(wav_files):
        print(f"处理文件: {file_path}")
        data_preemph, sample_rate, N = process_wav_file(
            file_path, duration, pre_emphasis=0.97
        )
        if data_preemph is None:
            continue

        time_bins, freq_bins, stft_matrix_lin = stft_custom(
            data_preemph, sample_rate, window_size, hop_size
        )
        stft_matrix_db = 20 * np.log10(stft_matrix_lin + 1e-10)

        file_name = os.path.basename(file_path)
        title = f"STFT 幅度谱 - {file_name}"

        # 绘制STFT
        if save_plots_stft:
            if output_dir_stft is None:
                output_dir_stft = os.path.join(directory_path, "stft_plots")
                os.makedirs(output_dir_stft, exist_ok=True)
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(
                time_bins,
                freq_bins,
                stft_matrix_db,
                shading="gouraud",
                cmap="plasma",
                vmin=-100,
                vmax=0,
            )
            plt.title(title)
            plt.ylabel("频率 (Hz)")
            plt.xlabel("时间 (秒)")
            plt.colorbar(label="幅度 (dB)")
            plt.ylim(0, sample_rate / 2)
            plot_path = os.path.join(output_dir_stft, f"{file_name}_stft.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"已保存STFT图表到: {plot_path}")
        else:
            # 如果不保存，直接显示
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(
                time_bins,
                freq_bins,
                stft_matrix_db,
                shading="gouraud",
                cmap="plasma",
                vmin=-100,
                vmax=0,
            )
            plt.title(title)
            plt.ylabel("频率 (Hz)")
            plt.xlabel("时间 (秒)")
            plt.colorbar(label="幅度 (dB)")
            plt.ylim(0, np.max(freq_bins))
            plt.show()

        # 生成Mel滤波器组
        mel_filter, bin_freqs = mel_filter_banks(
            sample_rate, window_size, n_mels, fmin, fmax
        )

        # 可视化Mel滤波器
        if visualize_mel_filters:
            plt.figure(figsize=(12, 6))
            for i in range(n_mels):
                plt.plot(bin_freqs, mel_filter[i], alpha=0.7)
            plt.title("Mel滤波器组频率响应")
            plt.xlabel("频率(Hz)")
            plt.ylabel("权重")
            plt.grid(True)
            plt.show()

        # 应用Mel滤波器
        mel_spectrogram_lin = np.dot(mel_filter, stft_matrix_lin)
        mel_spectrogram_db = 20 * np.log10(mel_spectrogram_lin + 1e-10)
        mel_freqs = np.arange(n_mels)
        mel_title = f"Mel 频谱图 - {file_name}"

        # 绘制Mel频谱图
        if save_plots_mel:
            if output_dir_mel is None:
                output_dir_mel = os.path.join(directory_path, "mel_plots")
                os.makedirs(output_dir_mel, exist_ok=True)
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(
                time_bins,
                mel_freqs,
                mel_spectrogram_db,
                shading="gouraud",
                cmap="magma",
                vmin=-100,
                vmax=0,
            )
            plt.title(mel_title)
            plt.ylabel("Mel 通道")
            plt.xlabel("时间 (秒)")
            plt.colorbar(label="幅度 (dB)")
            plot_path = os.path.join(output_dir_mel, f"{file_name}_mel.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"已保存Mel图表到: {plot_path}")
        else:
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(
                time_bins,
                mel_freqs,
                mel_spectrogram_db,
                shading="gouraud",
                cmap="magma",
                vmin=-100,
                vmax=0,
            )
            plt.title(mel_title)
            plt.ylabel("Mel 通道")
            plt.xlabel("时间 (秒)")
            plt.colorbar(label="幅度 (dB)")
            plt.show()

        # 计算MFCC

        # 1. 计算功率谱
        stft_power = stft_matrix_lin**2 / N

        # 2. 应用Mel滤波器
        mel_spectrogram_power = np.dot(mel_filter, stft_power)

        # 3. 进行DCT，提取MFCC特征
        mfcc = dct(mel_spectrogram_power, type=2, axis=0, norm="ortho")[:n_mfcc, :]

        # 4. 均值归一化（按时间轴归一化）
        mfcc -= np.mean(mfcc, axis=1, keepdims=True)

        mfcc_title = f"MFCC 特征图 - {file_name}"

        # 绘制MFCC特征图
        if save_plots_mfcc:
            if output_dir_mfcc is None:
                output_dir_mfcc = os.path.join(directory_path, "mfcc_plots")
                os.makedirs(output_dir_mfcc, exist_ok=True)
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(
                time_bins, np.arange(n_mfcc), mfcc, shading="gouraud", cmap="viridis"
            )
            plt.title(mfcc_title)
            plt.ylabel("MFCC 通道")
            plt.xlabel("时间 (秒)")
            plt.colorbar(label="MFCC系数值")
            plot_path = os.path.join(output_dir_mfcc, f"{file_name}_mfcc.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"已保存MFCC图表到: {plot_path}")
        else:
            plot_mfcc(time_bins, mfcc, mfcc_title)


if __name__ == "__main__":
    wav_directory = r".\data\raw\train"

    output_dir_mel = r".\data\pre_processed\mel"
    output_dir_stft = r".\data\pre_processed\stft"
    output_dir_mfcc = r".\data\pre_processed\mfcc"

    # 新增参数 visualize_mel_filters=True 来可视化mel滤波器
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
        visualize_mel_filters=False,  # 打开Mel滤波器可视化
        output_dir_stft=output_dir_stft,
        output_dir_mel=output_dir_mel,
        output_dir_mfcc=output_dir_mfcc,
    )
