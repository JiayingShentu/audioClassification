import matplotlib.pyplot as plt
import librosa
import librosa.display

def demo_plot():
    audio = 'D:\\毕设相关\\苏州训练\\音频\\铁芯松动电流\\2022-09-13-15-49.wav'
    y, sr = librosa.load(audio, sr=44100)
    y_ps = librosa.effects.pitch_shift(y, sr, n_steps=3)
    y_ts = librosa.effects.time_stretch(y, rate=1.1)
    plt.subplot(311)
    plt.plot(y)
    plt.title('Original waveform')
    plt.axis([0, 200000, -1, 1])
    plt.subplot(312)
    plt.plot(y_ps)
    plt.title('Pitch Shift transformed waveform')
    plt.axis([0, 200000, -1, 1])
    plt.subplot(313)
    plt.plot(y_ts)
    plt.title('Time Stretch transformed waveform')
    plt.axis([0, 200000, -1, 1])
    plt.tight_layout()
    plt.show()
 
demo_plot()
