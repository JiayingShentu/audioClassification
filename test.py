import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt

file_name='fold5/6508-9-0-1.wav'

audio_data,sampling_rate=librosa.load(file_name)
#librosa.display.waveplot(audio_data,sr=sampling_rate)
#ipd.Audio(file_name)
print(audio_data)
print(sampling_rate)

# audio_dataset_path='/content/'
# metadata=pd.read_csv('UrbanSound8K.csv')
# metadata.head()