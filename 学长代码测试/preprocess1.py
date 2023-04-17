from python_speech_features import mfcc
import numpy as np
import librosa
import os
from typing import List, Dict
import json
from tqdm import tqdm

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
    files = 'D:\\毕设相关\\苏州训练\\音频\\'

    # 获得标签文本和数字的对应
    label_list = []
    for file in os.listdir(files):
        label = file  # 标签的获取根据目录结构更改
        print(label)
        label_list.append(label)
    label_dict, num_dict = label_num_transform(label_list)
    print(label_dict,num_dict)

    # # 把音频转成mfcc形式的文件, 并将mfcc和标签保存为.npy文件
    # mfcc_path = 'mfccs.npy'
    # label_path = 'labels.npy'
    # for i, file in tqdm(enumerate(files)):
    #     signal, sr = librosa.load(file, sr=None, mono=True)
    #     signal_mfcc = get_mfcc(signal, sr, 0.4, 0.4)
    #     label = file.split(os.sep)[-2]
    #     label_num = np.ones(signal_mfcc.shape[0]) * label_dict[label]

    #     if i == 0:
    #         signal_mfccs = signal_mfcc
    #         label_nums = label_num
    #     else:
    #         signal_mfccs = np.concatenate([signal_mfccs, signal_mfcc], axis=0)
    #         label_nums = np.concatenate([label_nums, label_num])

    # np.save(mfcc_path, signal_mfccs)
    # np.save(label_path, label_nums)
    # print(signal_mfccs.shape)
    # print(label_nums.shape)
