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

            #######
            RNN、GRU
            #######
            Inputs：input,h_0
                 input：(seq_len, batch,input_size)
                 h_0：(num_layers * num_directions, batch, hidden_size)，初始状态，forward时不提供则默认为0。双向时 num_directions=2，否则为1
            Outputs：output,h_n
                 output:(seq_len, batch, num_directions * hidden_size),最后一层的所有h输出
                 h_n:(num_layers * num_directions, batch, hidden_size),最后一层t = seq_len的最后那个hidden state输出

            #######
            LSTM
            #######
            Inputs：input,(h_0,c_0)
                 input：(seq_len, batch,input_size),没区别
                 (h_0,c_0)： h_0,c_0都是(num_layers * num_directions, batch, hidden_size)，forward时不提供则默认为0。
            Outputs：output, (h_n, c_n)
                 output:(seq_len, batch, num_directions * hidden_size),最后一层的所有h输出
                 (h_n, c_n):h_n,c_n都是(num_layers * num_directions, batch, hidden_size),最后一层t = seq_len的hidden state和cell state输出
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
    # mfcc_path = 'mfccs.npy'
    # label_path = 'labels.npy'
    # mfccs = np.load(mfcc_path, allow_pickle=True)
    # labels = np.load(label_path, allow_pickle=True)
    # print(mfccs.shape, labels.shape)
    # train_data, other_data, train_label, other_label = model_selection.train_test_split(mfccs, labels, random_state=1,
    #                                                                                     train_size=0.7, test_size=0.3)
    # valid_data, test_data, valid_label, test_label = model_selection.train_test_split(other_data, other_label,
    #                                                                                   random_state=1, train_size=0.5,
    #                                                                                   test_size=0.5)
    train_data = np.load(r'学长代码测试\train_data.npy', allow_pickle=True)
    valid_data = np.load(r'学长代码测试\valid_data.npy', allow_pickle=True)
    test_data = np.load(r'学长代码测试\test_data.npy', allow_pickle=True)

    train_label = np.load(r'学长代码测试\train_label.npy', allow_pickle=True)
    valid_label = np.load(r'学长代码测试\valid_label.npy', allow_pickle=True)
    test_label = np.load(r'学长代码测试\test_label.npy', allow_pickle=True)

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
    train_data_tensor, valid_data_tensor, test_data_tensor = trans(train_data_tensor, valid_data_tensor,
                                                                   test_data_tensor)
    savepath = r"models/"
    RNN = RNNModule('GRU', 11, 26, 26, 1, add_linear=True)
    rnn_pre = train(RNN, train_data_tensor, train_labels_tensor, epochs=10, batch_size=4, clip=None,
                    X_val=valid_data_tensor,
                    savepath=savepath, Y_val=valid_labels_tensor, testmode=True)
    
    # random_forest = ensemble.RandomForestClassifier(random_state=0,n_estimators=1000)  
    # random_forest.fit(train_data_tensor, train_labels_tensor) 

    # prediction1=random_forest.predict(test_data_tensor)
    # print('随机森林')
    # print('预测前20个结果为：\n',prediction1[:20])
    # true1=np.sum(prediction1==test_labels_tensor)
    # print('预测对的结果数目为：', true1)
    # print('预测错的的结果数目为：', test_labels_tensor.shape[0]-true1)
    # print('预测结果准确率为：', true1/test_labels_tensor.shape[0])