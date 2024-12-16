import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
# 动态配置字体
import matplotlib

matplotlib.rcParams["font.family"] = "SimHei"  # 或其他中文字体
matplotlib.rcParams["axes.unicode_minus"] = False


# 读取.wav文件
file_path = r".\data\audio\1-100210-B-36.wav"
sample_rate, data = wavfile.read(file_path)

# 如果是立体声，选择一个通道
if len(data.shape) > 1:
    data = data[:, 0]

# FFT
duration = 5  # 秒
N = sample_rate * duration
fft_data = data[:N]

# 归一化
if np.max(np.abs(data)) == 0:
    print(f"文件 {file_path} 的数据全为零，跳过处理。")
fft_data = fft_data / np.max(np.abs(fft_data))


fft_result = np.fft.fft(fft_data)
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
plt.pcolormesh(t, f, 20 * np.log10(magnitude + 1e-10), shading="gouraud")
plt.title("STFT 幅度谱")
plt.ylabel("频率 (Hz)")
plt.xlabel("时间 (秒)")
plt.colorbar(label="幅度 (dB)")
plt.ylim(0, sample_rate / 2)
plt.show()
