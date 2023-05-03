import numpy as np
from python_speech_features import mfcc
import joblib
import librosa
import librosa.display
import os
import torch
import torch.nn as nn
import time
import json

class RNNModule(nn.Module):
    def __init__(self, rnn_type, num_classes, ninp, nhid, nlayers, batch_first=False, dropout=0.5, bidir=False,
                 add_linear=True):
        r"""
            args:
                 rnn_type:选择'RNN_TANH'，'RNN_RELU'，'LSTM', 'GRU'中的一种
                 ninp:输入x的 宽度/特征数
                 nhid:隐藏层h的 宽度/特征数
                 nlayers:模型有几层
                 batch_first:为True时输入和输出形式为(batch, seq, feature)，否则是(seq, batch, feature)形式  (batch_first为true的时候输出h好像不对？)
                 dropout:dropout概率，将对每层的输出进行dropout处理，除了最后一层。0为没有dropout
                 bidirectional:是否双向
                 add_linear:是否在最后加一层全连接层，使得hidden层维度不必与总类别数一致,默认True
        """
        super(RNNModule, self).__init__()
        if nlayers == 1:  # 一层时不需要dropout
            self.drop = nn.Dropout(0)
        else:
            self.drop = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.ninp = ninp
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.add_linear = add_linear
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, batch_first=batch_first, dropout=dropout,
                                             bidirectional=bidir)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, batch_first=batch_first, dropout=dropout,
                              bidirectional=bidir)
        if add_linear == True:
            # decoder层，用来把隐藏层输出维度转换到和类别数对应,这样nhid就可以不等于num_classes了
            self.decoder = nn.Linear(nhid, num_classes)
        else:
            # 否则检验nhid是否等于num_classes
            assert nhid == num_classes

    def forward(self, input):
        outputs, hidden = self.rnn(input)
        if self.add_linear == True:
            outputs = self.drop(outputs)
            outputs = self.decoder(outputs)
        return outputs, hidden  # (seq,batch,num_classes),(num_layers * num_directions, batch, nhid)

def read_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data
    
def standard_scaler(x):
    return (x - x.mean()) / x.std()

def batch_standard_scaler(x):
    for i in range(x.shape[0]):
        x[i] = standard_scaler(x[i])
    return x

def np_to_torch(self, np_array):
        return torch.from_numpy(np_array.astype(np.float32))

def trans(self, tensor):
        return torch.transpose(tensor, 0, 1)

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
    
if __name__ == '__main__':
    
    model_path = 'models\GRU_1_4.tjm'
    GRU = RNNModule('GRU', 11, 26, 26, 1, add_linear=True)
    GRU.load_state_dict(torch.load(model_path))
    model = GRU

    test_data = np.load(r'final_predata\test_data.npy', allow_pickle=True)
    test_label = np.load(r'final_predata\test_label.npy', allow_pickle=True)
    test_data = batch_standard_scaler(test_data)

    # signal_mfcc = np_to_torch(signal_mfcc)
    # signal_mfcc = trans(signal_mfcc)

    test_data_tensor = torch.tensor(test_data).float()
    test_labels_tensor = torch.tensor(test_label).long()
    X_test=test_data_tensor
    Y_test=test_labels_tensor

    print("test_data:",test_data)
    print("test_data_tensor:",test_data_tensor)
    print("test_labels_tensor:",test_labels_tensor)

    _, precision_test = test(model, X_test, Y_test)
    print("valid precision: {}%".format(str(precision_test * 100)))

    # model.eval()
    # output, h_out = model(test_data_tensor)
    # output = output[-1, :, :].squeeze()
    # _, predicted = torch.max(output, 1)
    # if test_labels_tensor == None:  # 没标签时仅返回预测结果
    #     return predicted
    # else:
    #     # 准确率计算
    #     accuracy = (predicted == Y_test).sum().item() / Y_test.size(0)
    #     return predicted, accuracy
    # print(prediction2)
    # prediction2 = np.array(prediction2).astype(np.int32)
    # num_to_label = read_from_json('num_dict.json')
    # result = num_to_label[str(np.argmax(prediction2))]
    # print(result)
    

    # model = get_model()
    # files = get_file_paths(r'..\音频')
    # for file in files:
    #     print(file, model.predict(file))
