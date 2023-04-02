import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble

#提取音频特征
def features_extractor(file):
    #file_name="D:/毕设相关/2021.5西电实验/data/wav格式2通道/电流/40.wav"
    file_name=file;
    sr=44100
    offset=0.1
    duration=5
    mono=True

    sample=librosa.load(file_name,sr,offset,duration, mono)[0]
    mfccs = librosa.feature.mfcc(y=sample, sr=sr, n_mfcc=40)#这里对音频信号做切割，并提取mfcc特征向量
    return mfccs

extracted_features=[]
filePath ='D:\\毕设相关\\2021.5西电实验\\data\\wav格式2通道\\'
for i in os.listdir(filePath):
    for j in os.listdir(filePath+i+"\\"):
        file_name = filePath+i+"\\"+j
        class_labels=i
        data=features_extractor(file_name)
        for data_slice in data:
            extracted_features.append([data_slice,class_labels])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
print(extracted_features_df.head(10))

print(extracted_features_df['class'].value_counts())

X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

random_forest = ensemble.RandomForestClassifier(random_state=0,n_estimators=10000)  
random_forest.fit(X_train, y_train) 

prediction1=random_forest.predict(X_test)
print('随机森林')
print('预测前20个结果为：\n',prediction1[:20])
true1=np.sum(prediction1==y_test)
print('预测对的结果数目为：', true1)
print('预测错的的结果数目为：', y_test.shape[0]-true1)
print('预测结果准确率为：', true1/y_test.shape[0])





