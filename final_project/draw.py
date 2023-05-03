import librosa
import numpy as np
import matplotlib.pyplot as plt
import random

def gauss_noisy(x):
    """
    对输入数据加入高斯噪声
    """
    x_add_noise=x
    mu = 0
    sigma = 0.001
    for i in range(len(x)):
        x_add_noise[i] += random.gauss(mu, sigma)
    return x_add_noise


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
fs = 16000

filePath="D:\\毕设相关\\苏州训练\\音频\\铁芯松动电流\\2022-09-13-15-49.wav"
wav_data, _ = librosa.load(filePath, sr=None, mono=True)

# ########### 画图
plt.subplot(2, 2, 1)
plt.title("语谱图", fontsize=15)
plt.specgram(wav_data, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.subplot(2, 2, 2)
plt.title("波形图", fontsize=15)
time = np.arange(0, len(wav_data)) * (1.0 / fs)
plt.plot(time, wav_data)
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('振幅', fontsize=15)

# plt.tight_layout()
# plt.show()

signal_add_noise=gauss_noisy(wav_data)

# ########### 画图
plt.subplot(2, 2, 3)
plt.title("语谱图", fontsize=15)
plt.specgram(signal_add_noise, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('频率/Hz', fontsize=15)

plt.subplot(2, 2, 4)
plt.title("波形图", fontsize=15)
time = np.arange(0, len(signal_add_noise)) * (1.0 / fs)
plt.plot(time, signal_add_noise)
plt.xlabel('秒/s', fontsize=15)
plt.ylabel('振幅', fontsize=15)

plt.tight_layout()
plt.show()