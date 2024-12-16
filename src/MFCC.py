import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import glob
import librosa  # 新增依赖
import librosa.display  # 用于后续可视化(可选)

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

    # 位反转排序
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

    # 蝶形运算
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
    """
    计算大于等于 x 的最小 2 的幂
    """
    return 1 << (x - 1).bit_length()


def pad_to_power_of_two(x):
    """
    将输入序列填充为 2 的幂长度
    """
    N = len(x)
    N_padded = next_power_of_two(N)
    if N_padded == N:
        return x
    else:
        return np.pad(x, (0, N_padded - N), "constant")


def hann_window(N):
    """
    生成汉宁窗
    """
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)


def plot_fft(freq, magnitude, title):
    """
    绘制 FFT 频谱图
    """
    plt.figure(figsize=(12, 6))
    plt.plot(freq, magnitude)
    plt.title(title)
    plt.xlabel("频率 (Hz)")
    plt.ylabel("幅度")
    plt.grid()
    plt.show()


def plot_stft(time_bins, freq_bins, magnitude_db, title, cmap="plasma", vmin=-100, vmax=0):
    """
    绘制 STFT 频谱图（时频图），幅度为 dB 尺度
    """
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


def plot_mel_spectrogram(time_bins, mel_freqs, mel_db, title, cmap='magma', vmin=-100, vmax=0):
    """
    绘制 Mel 频谱图（时频图），幅度为 dB 尺度
    mel_freqs: Mel 滤波器组数目对应的 Mel 频率轴（这里直接使用滤波器组个数作为频率轴刻度）
    """
    plt.figure(figsize=(12, 6))
    # mel_db shape: (n_mels, time_frames)
    plt.pcolormesh(time_bins, mel_freqs, mel_db, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.ylabel("Mel 频率通道")
    plt.xlabel("时间 (秒)")
    plt.colorbar(label="幅度 (dB)")
    plt.show()


def process_wav_file(file_path, duration=5):
    """
    读取并处理单个 .wav 文件，计算 FFT 幅度谱
    """
    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        print(f"无法读取文件 {file_path}，错误: {e}")
        return None, None

    # 如果是立体声，选择一个通道
    if len(data.shape) > 1:
        data = data[:, 0]

    # 选择前 duration 秒的数据
    N = int(sample_rate * duration)
    if N > len(data):
        print(
            f"文件 {file_path} 长度不足 {duration} 秒，实际长度: {len(data)/sample_rate:.2f} 秒"
        )
        N = len(data)
    data = data[:N]

    # 归一化
    if np.max(np.abs(data)) == 0:
        print(f"文件 {file_path} 的数据全为零，跳过处理。")
        return None, None
    data = data / np.max(np.abs(data))

    # 填充为 2 的幂长度
    data_padded = pad_to_power_of_two(data)

    # 进行 FFT
    X = fft_iterative(data_padded)

    # 计算频率轴
    N_padded = len(data_padded)
    freq = np.fft.fftfreq(N_padded, d=1 / sample_rate)

    # 只取正频率部分
    pos_mask = freq >= 0
    freq = freq[pos_mask]
    X = X[pos_mask]

    # 计算幅度谱并归一化
    magnitude = np.abs(X) * 2 / N_padded

    return freq, magnitude


def stft_custom(x, sample_rate, window_size=1024, hop_size=512):
    """
    自定义实现的短时傅里叶变换（STFT）
    返回线性幅度谱（未转换为 dB）
    """
    window = hann_window(window_size)
    num_frames = 1 + (len(x) - window_size) // hop_size
    # stft_matrix 用来存线性幅度谱 (非 dB)
    stft_matrix = np.zeros((window_size // 2 + 1, num_frames), dtype=np.float32)
    time_bins = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        frame = x[start:end]
        if len(frame) < window_size:
            # 数据长度不够，用0填充
            frame = np.pad(frame, (0, window_size - len(frame)), 'constant')
        frame_win = frame * window
        # 填充为 2 的幂长度
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
    n_mels=128,  # 新增Mel滤波器数目参数
    fmin=0,
    fmax=None,
    save_plots=False,
    output_dir=None,
):
    """
    批量处理目录下的所有 .wav 文件，计算 STFT 幅度谱并Mel频谱图
    """
    wav_files = glob.glob(os.path.join(directory_path, "*.wav"))
    if not wav_files:
        print(f"在目录 {directory_path} 中未找到任何 .wav 文件。")
        return

    if save_plots:
        if output_dir is None:
            output_dir = os.path.join(directory_path, "stft_plots")
        os.makedirs(output_dir, exist_ok=True)

    for file_path in wav_files:
        print(f"处理文件: {file_path}")
        try:
            sample_rate, data = wavfile.read(file_path)
        except Exception as e:
            print(f"无法读取文件 {file_path}，错误: {e}")
            continue

        # 如果是立体声，选择一个通道
        if len(data.shape) > 1:
            data = data[:, 0]

        N = int(sample_rate * duration)
        if N > len(data):
            print(f"文件 {file_path} 长度不足 {duration} 秒，实际长度: {len(data)/sample_rate:.2f} 秒")
            N = len(data)
        data = data[:N]

        if np.max(np.abs(data)) == 0:
            print(f"文件 {file_path} 的数据全为零，跳过处理。")
            continue
        data = data / np.max(np.abs(data))

        # 计算STFT的线性幅度谱
        time_bins, freq_bins, stft_matrix_lin = stft_custom(data, sample_rate, window_size, hop_size)

        # 转换为分贝尺度的 STFT
        stft_matrix_db = 20 * np.log10(stft_matrix_lin + 1e-10)

        # 绘制STFT
        file_name = os.path.basename(file_path)
        title = f"STFT 幅度谱 - {file_name}"

        if save_plots:
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
            plot_path = os.path.join(output_dir, f"{file_name}_stft.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"已保存图表到: {plot_path}")
        else:
            plot_stft(time_bins, freq_bins, stft_matrix_db, title)

        # ==== 生成Mel滤波器组并计算Mel频谱图 ====
        # 若fmax为None，则默认为 sample_rate/2
        if fmax is None:
            fmax = sample_rate / 2

        # 生成 Mel 滤波器组
        mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=n_mels, fmin=fmin, fmax=fmax)

        # stft_matrix_lin: shape (freq_bins, time_frames)
        # mel_filter: shape (n_mels, freq_bins)
        # 首先确认维度匹配，如果 mel_filter 对应的freq_bins与 stft的freq_bins点数不一致，需要做适配。
        # 这里n_fft=window_size时，freq_bins大小为 window_size//2+1，与mel_filter期望的一致。

        # 应用Mel滤波器(使用线性幅度谱)
        mel_spectrogram = np.dot(mel_filter, stft_matrix_lin)

        # 转换为dB
        mel_spectrogram_db = 20 * np.log10(mel_spectrogram + 1e-10)

        # 时间轴与STFT相同
        # mel频率轴只是 n_mels 个mel通道，不对应真实频率，可直接使用 np.arange(n_mels)
        mel_freqs = np.arange(n_mels)

        mel_title = f"Mel 频谱图 - {file_name}"
        if save_plots:
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(
                time_bins,
                mel_freqs,
                mel_spectrogram_db,
                shading='gouraud',
                cmap='magma',
                vmin=-100,
                vmax=0
            )
            plt.title(mel_title)
            plt.ylabel("Mel 通道")
            plt.xlabel("时间 (秒)")
            plt.colorbar(label="幅度 (dB)")
            plot_path = os.path.join(output_dir, f"{file_name}_mel.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"已保存Mel图表到: {plot_path}")
        else:
            plot_mel_spectrogram(time_bins, mel_freqs, mel_spectrogram_db, mel_title)


if __name__ == "__main__":
    # 指定包含 .wav 文件的目录路径
    wav_directory = r".\data\audio"  # 请替换为你的目录路径

    # 增加Mel谱图绘制，新增参数 n_mels, fmin, fmax，可根据需要调整
    batch_stft_processing(
        directory_path=wav_directory,
        duration=5,
        window_size=1024,
        hop_size=512,
        n_mels=128,
        fmin=0,
        fmax=None,
        save_plots=False,
    )
