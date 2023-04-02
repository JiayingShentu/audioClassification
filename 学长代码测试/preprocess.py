from python_speech_features import mfcc
import numpy as np
import librosa
import os
from typing import List, Dict
import json
from tqdm import tqdm


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


def get_file_paths(dir) -> List[str]:
    """
    获得目录下所有文件路径的列表
    :param dir: 目录路径
    :return: 包含文件路径的列表
    """
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


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


def write_to_json(path, mydict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(mydict, f, ensure_ascii=False, indent=4)


def read_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data


if __name__ == '__main__':
    files = get_file_paths('音频')

    # 获得标签文本和数字的对应
    label_dict_path = 'label_dict.json'
    num_dict_path = 'num_dict.json'
    if not os.path.exists(label_dict_path) or not os.path.exists(num_dict_path):
        label_list = []
        for file in files:
            label = file.split(os.sep)[-2]  # 标签的获取根据目录结构更改
            label_list.append(label)
        label_dict, num_dict = label_num_transform(label_list)
        write_to_json(label_dict_path, label_dict)
        write_to_json(num_dict_path, num_dict)

    else:
        label_dict = read_from_json(label_dict_path)
        num_dict = read_from_json(num_dict_path)

    # print(label_dict, num_dict)

    # 把音频转成mfcc形式的文件, 并将mfcc和标签保存为.npy文件
    mfcc_path = 'mfccs.npy'
    label_path = 'labels.npy'
    for i, file in tqdm(enumerate(files)):
        signal, sr = librosa.load(file, sr=None, mono=True)
        signal_mfcc = get_mfcc(signal, sr, 0.4, 0.4)
        label = file.split(os.sep)[-2]
        label_num = np.ones(signal_mfcc.shape[0]) * label_dict[label]

        if i == 0:
            signal_mfccs = signal_mfcc
            label_nums = label_num
        else:
            signal_mfccs = np.concatenate([signal_mfccs, signal_mfcc], axis=0)
            label_nums = np.concatenate([label_nums, label_num])

    np.save(mfcc_path, signal_mfccs)
    np.save(label_path, label_nums)
    print(signal_mfccs.shape)
    print(label_nums.shape)



