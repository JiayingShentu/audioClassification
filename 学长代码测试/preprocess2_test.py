from python_speech_features import mfcc
import numpy as np
import librosa
import os
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim
import time
from sklearn import model_selection
import os
import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import ensemble

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

def train(RNNModule, X_train, Y_train, epochs, batch_size, savepath=None, clip=0.25, X_val=None, Y_val=None,
          testmode=False):
    r"""
    args:
        RNNModule:model
        X_train:训练集，(seq, batch,feature)
        Y_train:标签，(batch,)，不需要经过one_hot编码
        epochs:完整训练几轮
        batch_size:包大小
        savepath:保存路径，例如r"D:\文件归档\记录\2020-9-29-变压器过励磁过电流与紧固件松动\data\RNN.tjm"
        clip:gradient clipping,None时不使用
    """
    # Turn on training mode which enables dropout.
    model = RNNModule
    model.train()
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    filePath = epochPath = savepath + model.rnn_type + '_' + str(model.nlayers) + '_' + str(batch_size) + '.tjm'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    size = X_train.size()[1]
    print('训练集总大小:', size)

    start_time = time.time()  # 记录开始时刻
    precision = []
    for epoch in range(epochs):
        epochPath = savepath + model.rnn_type + '_' + str(model.nlayers) + '_' + str(batch_size) + '_' + str(
            epoch) + '.tjm'
        if testmode:  # 用来画训练过程的图
            _, tmp = test(model, X_val, Y_val)
            # print(epoch,':',tmp)
            precision.append(tmp)

        running_loss = 0.0
        i, index = 0, 0  # 记录第几个batch,当前索引
        while index < size:
            # 得到输入和标签
            inputs = X_train[:, index:index + batch_size, :]
            labels = Y_train[index:index + batch_size]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, h_out = model(inputs)
            d0, d1, d2 = outputs.size()  # 用squeeze好像会出问题
            output = outputs[-1, :, :].view(d1, d2)
            loss = criterion(output, labels)
            loss.backward()

            if clip != None:
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000 / batch_size))
                running_loss = 0.0
            # 更新i和index
            i += 1
            index += batch_size
        if epoch % 1 == 0:
            if X_val != None and Y_val != None:
                _, precision_test = test(model, X_val, Y_val)
                print("valid precision: {}%".format(str(precision_test * 100)))
            torch.save(model.state_dict(), epochPath)

    # 保存模型
    if savepath != None:
        torch.save(model.state_dict(), filePath)

    end_time = time.time()
    time_cost = end_time - start_time
    print('Finished Training,总耗时为(s):', time_cost)
    return precision

def standard_scaler(x):
    return (x - x.mean()) / x.std()

def batch_standard_scaler(x):
    for i in range(x.shape[0]):
        x[i] = standard_scaler(x[i])
    return x

def test(RNNModule, X_test, Y_test=None, savepath=None):
    r"""
    args:
        RNNModule:model
        X_test:训练集，(seq, batch,feature)
        Y_test:标签，(batch,)，不需要经过one_hot编码
        epochs:完整训练几轮
        batch_size:包大小
        savepath:保存路径，例如r"D:\文件归档\记录\2020-9-29-变压器过励磁过电流与紧固件松动\data\RNN.tjm"
    """
    # Turn on evaluation mode which disables dropout.
    model = RNNModule
    model.eval()

    output, h_out = model(X_test)
    output = output[-1, :, :].squeeze()
    _, predicted = torch.max(output, 1)
    if Y_test == None:  # 没标签时仅返回预测结果
        return predicted
    else:
        # 准确率计算
        accuracy = (predicted == Y_test).sum().item() / Y_test.size(0)
        return predicted, accuracy

def trans(*tensors):
    return [torch.transpose(i, 0, 1) for i in tensors] 

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
    extracted_features=np.array([])
    for i in os.listdir(files):
        for file in os.listdir(files+i+"\\"):
            filePath=files+i+"\\"+file
            signal, sr = librosa.load(filePath, sr=None, mono=True)
            signal_mfcc = get_mfcc_plus(signal, sr, 0.4, 0.4)
            label=i
            label_num=np.ones(signal_mfcc.shape[0]) * label_dict[label]
            #print(i,file,label,label_num)

            if num == 0:
                signal_mfccs =signal_mfcc
                label_nums = label_num
            else:
                signal_mfccs = np.concatenate([signal_mfccs, signal_mfcc], axis=0)
                label_nums = np.concatenate([label_nums, label_num])
            num=num+1

    #print(signal_mfccs,label_nums)

    np.save(r'D:\毕设相关\audioClassification\学长代码测试\mfccs.npy', signal_mfccs)
    np.save(r'D:\毕设相关\audioClassification\学长代码测试\labels.npy', label_nums)
    print(signal_mfccs)
    print(label_nums)
    mfccs=signal_mfccs
    labels=label_nums

    train_data, other_data, train_label, other_label = model_selection.train_test_split(mfccs, labels, random_state=1,
                                                                                        train_size=0.7, test_size=0.3)
    valid_data, test_data, valid_label, test_label = model_selection.train_test_split(other_data, other_label,
                                                                                      random_state=1, train_size=0.5,
                                                                                      test_size=0.5)
    # 对mfcc进行标准化处理
    train_data = batch_standard_scaler(train_data)
    valid_data = batch_standard_scaler(valid_data)
    test_data = batch_standard_scaler(test_data)

    train_data_tensor = torch.tensor(train_data).float()
    train_labels_tensor = torch.tensor(train_label).long()
    valid_data_tensor = torch.tensor(valid_data).float()
    valid_labels_tensor = torch.tensor(valid_label).long()
    test_data_tensor = torch.tensor(test_data).float()
    test_labels_tensor = torch.tensor(test_label).long()
    # train_data_tensor, valid_data_tensor, test_data_tensor = trans(train_data_tensor, valid_data_tensor,
    #                                                                test_data_tensor)
    

    random_forest = ensemble.RandomForestClassifier(random_state=0,n_estimators=1000)  
    random_forest.fit(train_data_tensor, train_labels_tensor) 

    prediction1=random_forest.predict(test_data_tensor)
    print('随机森林')
    print('预测前20个结果为：\n',prediction1[:20])
    true1=np.sum(prediction1==test_labels_tensor)
    print('预测对的结果数目为：', true1)
    print('预测错的的结果数目为：', test_labels_tensor.shape[0]-true1)
    print('预测结果准确率为：', true1/test_labels_tensor.shape[0])

    