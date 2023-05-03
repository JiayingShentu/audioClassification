from python_speech_features import mfcc
import numpy as np
import librosa
import os
from typing import List, Dict
from sklearn import model_selection
from sklearn.model_selection import train_test_split

def get_mfcc(signal, fs, duration, step):
    # 输入 signal:原始音频信号，一维numpy数组
    #     fs:输入音频信号的采样频率
    #     duration:每帧时长，单位s  建议与mfcc配合
    #     step:帧移,百分比小数
    winlen = 0.04  # mfcc窗时长，40ms帧长
    winstep = 0.02  # mfcc相邻窗step，20ms帧移
    ori_mfcc = mfcc(signal, numcep=26, nfilt=52, samplerate=fs, winlen=winlen, winstep=winstep, nfft=2048, lowfreq=50,
                    highfreq=20000,
                    preemph=0, appendEnergy=False)  # 注意nfft点数与帧长的配合建议nfft=winlen*fs
    pick_num = int((duration - winlen) / winstep + 1)  # 取几个
    step_num = int(duration * step / winstep)  # 下一样本往后挪几个
    ind = pick_num
    max_ind = len(ori_mfcc)
    out_mfcc = []
    while ind <= max_ind:
        out_mfcc.append(ori_mfcc[ind - pick_num:ind])
        ind += step_num
    return np.array(out_mfcc)

def get_mfcc_plus(signal, fs, duration, step):
    # 输入 signal:原始音频信号，一维numpy数组
    #     fs:输入音频信号的采样频率
    #     duration:每帧时长，单位s  建议与mfcc配合
    #     step:帧移,百分比小数
    winlen = 0.04  # mfcc窗时长，40ms帧长
    winstep = 0.02  # mfcc相邻窗step，20ms帧移
    ori_mfcc = mfcc(signal, numcep=26, nfilt=52, samplerate=fs, winlen=winlen, winstep=winstep, nfft=2048, lowfreq=50,
                    highfreq=20000,
                    preemph=0, appendEnergy=False)  # 注意nfft点数与帧长的配合建议nfft=winlen*fs
    return np.array(ori_mfcc)

def label_num_transform(label_list: List) -> Dict:
    """
    将标签和数字相互转换
    :param label_list: 标签列表
    :return: {标签：数字}形式的字典 和 {数字：标签}形式的字典
    """
    labels = set(label_list)
    label_dict = dict(zip(labels, range(len(labels))))
    num_dict = {}
    for key, value in label_dict.items():
        num_dict[value] = key
    return label_dict, num_dict
        

if __name__ == '__main__':
    files = 'D:\\毕设相关\\苏州训练\\音频\\'

    # 获得标签文本和数字的对应
    label_list = []
    for file in os.listdir(files):
        label = file  # 标签的获取根据目录结构更改
        label_list.append(label)
    label_dict, num_dict = label_num_transform(label_list)
    print(label_dict,num_dict)

    # 把音频转成mfcc形式的文件, 并将mfcc和标签保存为.npy文件
    mfcc_path = 'mfccs.npy'
    label_path = 'labels.npy'
    num=0 # 统计音频数量

    for i in os.listdir(files):
        for file in os.listdir(files+i+"\\"):
            filePath=files+i+"\\"+file
            signal, sr = librosa.load(filePath, sr=None, mono=True)
            signal_mfcc = get_mfcc(signal, sr, 0.4, 0.4)
            label=i
            label_num=np.ones(signal_mfcc.shape[0]) * label_dict[label]
            #print(i,file,label,label_num)

            if num == 0:
                signal_mfccs = signal_mfcc
                label_nums = label_num
            else:
                signal_mfccs = np.concatenate([signal_mfccs, signal_mfcc], axis=0)
                label_nums = np.concatenate([label_nums, label_num])
            num=num+1

    #print(signal_mfccs,label_nums)

    # np.save(r'D:\毕设相关\audioClassification\学长代码测试\mfccs.npy', signal_mfccs)
    # np.save(r'D:\毕设相关\audioClassification\学长代码测试\labels.npy', label_nums)
    # print(signal_mfccs)
    # print(label_nums)

    train_data, other_data, train_label, other_label = model_selection.train_test_split(signal_mfccs, label_nums, random_state=1,
                                                                                        train_size=0.7, test_size=0.3)
    valid_data, test_data, valid_label, test_label = model_selection.train_test_split(other_data, other_label,
                                                                                      random_state=1, train_size=0.5,
                                                                                      test_size=0.5)

    np.save(r'D:\毕设相关\audioClassification\学长代码测试\data\train_data.npy', train_data)
    np.save(r'D:\毕设相关\audioClassification\学长代码测试\data\train_label.npy', train_label)
    np.save(r'D:\毕设相关\audioClassification\学长代码测试\data\valid_data.npy', valid_data)
    np.save(r'D:\毕设相关\audioClassification\学长代码测试\data\valid_label.npy', valid_label)
    np.save(r'D:\毕设相关\audioClassification\学长代码测试\data\test_data.npy', test_data)
    np.save(r'D:\毕设相关\audioClassification\学长代码测试\data\test_label.npy', test_label)

    