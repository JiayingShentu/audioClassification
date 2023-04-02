from python_speech_features import mfcc
import numpy as np
import librosa
import os
from typing import List, Dict
import json
from tqdm import tqdm

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

if __name__ == '__main__':
    files = get_file_paths('音频')
    print(files)

    # # 获得标签文本和数字的对应
    # label_dict_path = 'label_dict.json'
    # num_dict_path = 'num_dict.json'
    # if not os.path.exists(label_dict_path) or not os.path.exists(num_dict_path):
    #     label_list = []
    #     for file in files:
    #         label = file.split(os.sep)[-2]  # 标签的获取根据目录结构更改
    #         label_list.append(label)
    #     label_dict, num_dict = label_num_transform(label_list)
    #     write_to_json(label_dict_path, label_dict)
    #     write_to_json(num_dict_path, num_dict)

    # else:
    #     label_dict = read_from_json(label_dict_path)
    #     num_dict = read_from_json(num_dict_path)

    # # print(label_dict, num_dict)

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
