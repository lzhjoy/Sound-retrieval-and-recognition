import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import glob
import librosa  # 用于Mel滤波器和MFCC的计算
import librosa.display  # 可选，用于更方便的可视化
from tqdm import tqdm


# 动态配置字体
import matplotlib

matplotlib.rcParams["font.family"] = "SimHei"  # 或其他中文字体
matplotlib.rcParams["axes.unicode_minus"] = False


def fft_iterative(x):
    """
    迭代实现的 Cooley-Turkey FFT 算法
    """
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


def plot_fft(freq, magnitude, title):
    plt.figure(figsize=(12, 6))
    plt.plot(freq, magnitude)
    plt.title(title)
    plt.xlabel("频率 (Hz)")
    plt.ylabel("幅度")
    plt.grid()
    plt.show()


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
    """
    绘制 MFCC 特征图
    mfcc.shape = (n_mfcc, time_frames)
    """
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(
        time_bins, np.arange(mfcc.shape[0]), mfcc, shading="gouraud", cmap=cmap
    )
    plt.title(title)
    plt.ylabel("MFCC 通道")
    plt.xlabel("时间 (秒)")
    plt.colorbar(label="MFCC系数值")
    plt.show()


def process_wav_file(file_path, duration=5, pre_emphasis=0.97):
    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        print(f"无法读取文件 {file_path}，错误: {e}")
        return None, None

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
        return None, None
    data = data / np.max(np.abs(data))

    # 预加重
    data_preemph = np.append(data[0], data[1:] - pre_emphasis * data[:-1])

    data_padded = pad_to_power_of_two(data_preemph)

    X = fft_iterative(data_padded)
    N_padded = len(data_padded)
    freq = np.fft.fftfreq(N_padded, d=1 / sample_rate)

    pos_mask = freq >= 0
    freq = freq[pos_mask]
    X = X[pos_mask]

    magnitude = np.abs(X) * 2 / N_padded

    return freq, magnitude


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


def batch_stft_processing(
    directory_path,
    duration=5,
    window_size=1024,
    hop_size=512,
    n_mels=128,
    fmin=0,
    fmax=None,
    n_mfcc=13,  # 新增MFCC通道数（通常13）
    save_plots_stft=False,
    save_plots_mel=False,
    save_plots_mfcc=False,
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
        try:
            sample_rate, data = wavfile.read(file_path)
        except Exception as e:
            print(f"无法读取文件 {file_path}，错误: {e}")
            continue

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
            continue
        data = data / np.max(np.abs(data))

        time_bins, freq_bins, stft_matrix_lin = stft_custom(
            data, sample_rate, window_size, hop_size
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
            plot_stft(time_bins, freq_bins, stft_matrix_db, title)

        # Mel滤波器组计算
        if fmax is None:
            fmax = sample_rate / 2
        mel_filter = librosa.filters.mel(
            sr=sample_rate, n_fft=window_size, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        mel_spectrogram = np.dot(mel_filter, stft_matrix_lin)
        mel_spectrogram_db = 20 * np.log10(mel_spectrogram + 1e-10)
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
            plot_mel_spectrogram(time_bins, mel_freqs, mel_spectrogram_db, mel_title)

        # ==== 计算并绘制MFCC特征图 ====
        # MFCC通常从Mel功率谱获取。mel_spectrogram_db是dB值，我们需要转换回功率谱。
        # librosa的mfcc函数期望输入S为功率谱（非dB），因此需使用db_to_power将dB值转换为功率谱。
        mel_power = librosa.db_to_power(mel_spectrogram_db)
        mfcc = librosa.feature.mfcc(S=mel_power, sr=sample_rate, n_mfcc=n_mfcc)
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

    batch_stft_processing(
        directory_path=wav_directory,
        duration=5,
        window_size=1024,
        hop_size=512,
        n_mels=128,
        fmin=0,
        fmax=None,
        n_mfcc=13,  # 增加MFCC特征数
        save_plots_stft=True,
        save_plots_mel=True,
        save_plots_mfcc=True,
        output_dir_stft=output_dir_stft,
        output_dir_mel=output_dir_mel,
        output_dir_mfcc=output_dir_mfcc,
    )
