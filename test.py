import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from python_speech_features import mfcc

def frame_split(signal, duration, fs, step):#
    # 输入 signal:原始音频信号，一维numpy数组
    # duration:每帧时长，单位s  建议与mfcc配合
    # fs:输入音频信号的采样频率
    # step:帧移,百分比小数
    # 输出 out:二维矩阵，axis=0方向是帧数，末尾不足以组成一帧的舍去
    length = int(duration * fs)  # 每帧采样点数
    step = int(step * length)  # 帧移(按采样点数计)
    print("length:",length,",step:",step)
    if signal.shape[0] < length:
        print('长度过短，不足以组成一帧,已沿1轴重复')
        signal = np.tile(signal, (length // signal.shape[0] + 1))
    index = length
    num = 0  # 用来累积帧数
    while index <= signal.shape[0]:
        if num == 0:  # 第一次时特殊处理
            out = signal[index - length:index]
        else:  # 后续直接vstack
            out = np.vstack((out, signal[index - length:index]))  # 取出一帧
        index += step
        num += 1
    return out


file_name="D:/毕设相关/2021.5西电实验/data/wav格式2通道/电流/40.wav"
sr=44100
offset=0.1
duration=5
mono=True

sample=librosa.load(file_name,sr,offset,duration, mono=True)[0]
print("sample:",sample)
print("sample.shape:",sample.shape)
# audio_data, sampling_rate = librosa.load(file_name,sr=44100,offset=0.1,duration=5, mono=True)
# print("audio_data:",audio_data)
# print("audio_data.shape:",audio_data.shape)
# print("sampling_rate:",sampling_rate)
#如果y.shape样式为(2，n)则是立体声，如果是(n，)则是单声道

# mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
# print(mfccs)
signal_frames = frame_split(sample, 0.4, sr, 0.4)  # 400ms帧长，40%重叠率
print("signal_frames:",signal_frames)
signal_mfcc = np.array([mfcc(frame,numcep=26,nfilt=52,samplerate=sr,winlen=0.04,winstep=0.02,nfft=2048,
                lowfreq=50,highfreq=20000,preemph=0,appendEnergy=False)
                    for frame in signal_frames])  # 40ms帧长，20ms帧移  注意nfft点数与帧长的配合建议nfft=winlen*fs 
print("signal_mfcc:",signal_mfcc)