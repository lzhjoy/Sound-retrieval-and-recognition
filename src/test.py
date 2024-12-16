import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
import matplotlib
import librosa  # 用于生成mel滤波器组
import os

# 动态配置字体
matplotlib.rcParams["font.family"] = "SimHei"  # 或其他中文字体
matplotlib.rcParams["axes.unicode_minus"] = False

# 读取.wav文件
file_path = r".\data\audio\1-100210-B-36.wav"  # 请替换为你的音频文件路径
sample_rate, data = wavfile.read(file_path)

# 如果是立体声，选择一个通道
if len(data.shape) > 1:
    data = data[:, 0]

duration = 5  # 秒
N = sample_rate * duration
data = data[:N]

# 归一化
if np.max(np.abs(data)) == 0:
    print(f"文件 {file_path} 的数据全为零，跳过处理。")
data = data / np.max(np.abs(data))

# FFT
fft_result = np.fft.fft(data)
fft_freq = np.fft.fftfreq(N, d=1 / sample_rate)
positive_freqs = fft_freq[: N // 2]
positive_magnitude = np.abs(fft_result[: N // 2]) * (2 / N)

plt.figure(figsize=(12, 6))
plt.plot(positive_freqs, positive_magnitude)
plt.title("FFT 频谱")
plt.xlabel("频率 (Hz)")
plt.ylabel("幅度")
plt.grid()
plt.show()

# STFT 使用 scipy
nperseg = 1024
noverlap = nperseg // 2
f, t, Zxx = stft(
    data, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=noverlap
)
magnitude = np.abs(Zxx)

plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 20 * np.log10(magnitude + 1e-10), shading="gouraud", cmap="plasma")
plt.title("STFT 幅度谱")
plt.ylabel("频率 (Hz)")
plt.xlabel("时间 (秒)")
plt.colorbar(label="幅度 (dB)")
plt.ylim(0, sample_rate / 2)
plt.show()

# ============ 生成Mel频谱图 ============
# Mel滤波器参数
n_mels = 128  # Mel滤波器组的数量
fmin = 0
fmax = sample_rate / 2

# 利用 librosa.filters.mel 生成 Mel 滤波器组
mel_filter = librosa.filters.mel(
    sr=sample_rate, n_fft=nperseg, n_mels=n_mels, fmin=fmin, fmax=fmax
)

# magnitude.shape = (len(f), len(t))
# mel_filter.shape = (n_mels, n_fft//2+1) = (n_mels, nperseg//2+1)

# 确保频率维度匹配，如果采用 scipy.stft，n_fft = nperseg，会产生 (nperseg//2+1) 个频率点
# 这与 mel_filter 期望输入维度一致

# 将线性幅度谱映射到 Mel 频率轴
mel_spectrogram = np.dot(mel_filter, magnitude)

# 转换为dB
mel_spectrogram_db = 20 * np.log10(mel_spectrogram + 1e-10)

# 绘制Mel频谱图
plt.figure(figsize=(12, 6))
plt.pcolormesh(
    t,
    np.arange(n_mels),
    mel_spectrogram_db,
    shading="gouraud",
    cmap="magma",
    vmin=-100,
    vmax=0,
)
plt.title("Mel 频谱图")
plt.ylabel("Mel 频率通道")
plt.xlabel("时间 (秒)")
plt.colorbar(label="幅度 (dB)")
plt.show()
