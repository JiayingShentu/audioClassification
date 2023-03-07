import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt

file_name="D:/毕设相关/2021.5西电实验/data/wav格式2通道/电流/40.wav"

audio_data,sampling_rate=librosa.load(file_name)
librosa.display.waveplot(audio_data,sr=sampling_rate)
#ipd.Audio(file_name)
print(audio_data)
print(sampling_rate)

# audio_dataset_path='/content/'
# metadata=pd.read_csv('UrbanSound8K.csv')
# metadata.head()