# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:06:27 2020

技术分析三大假设：
（1）市场行为包容消化一切影响价格的任何因素：基本面、政治因素、心理因素等等因素都要最终通过买卖反映在价格中
（2）价格以趋势方式演变：对于已经形成的趋势来讲，通常是沿现存趋势继续演变
（3）历史会重演：技术分析和市场行为学与人类心理学有一定关系，价格形态通过特定的图表表示了人们对某市场看好或看淡的心理

搭建模型时的补充假设:
（1）历史数据中同时包含随机性与趋势性，其中，趋势性可预测，而随机性不可预测
（2）历史数据中同时包含影响后续走势的关键特征以及无关噪声，两者均与时间序列的长度呈正相关(特征过少会导致模型的欠拟合，噪声过多会导致模型的过拟合)
（3）对于短期趋势而言，历史数据点之间所相隔的时间越小，两者相关性越大

基于以上假设，当前设计理念为：以沪深300指数的60分钟线为训练数据，输入60个交易时（15个交易日）的走势，预测接下来20个交易时（5个交易日）的走势

接下来可尝试：
（1）输入m个交易日的走势预测接下来n个交易日的走势（10≤m≤60，5≤n≤15，2n≤m≤4n）
（2）输入m个交易日的涨跌幅预测接下来n个交易日的走势（增加当前点位特征）
（3）增加或改变特征维度
（4）更改模型结构或网络连接方式
（5）测试不同基金的表现

@author: TennyXu
"""
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data=pd.read_excel('000300.xlsx')
scaler = preprocessing.StandardScaler().fit(data.iloc[:,1:9])
data_t=scaler.transform(data.iloc[:,1:9])
#print(scaler.mean_)
#print(scaler.scale_ )
#dir(scaler)
def dataset_maker(data_t):
    n=1800
    seq_data=np.zeros([n,60,8])
    res_data=np.zeros([n,20])
    for i in range(n):
        seq_data[i,:,:]=data_t[i:i+60,0:9]
        res_data[i,:]=data_t[i+60:i+80,3]
    return seq_data,res_data

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))


data_dim = 8
timesteps = 60

model = Sequential()
model.add(LSTM(256, return_sequences=True,input_shape=(timesteps, data_dim)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(20, activation=None))

model.compile(loss='mse',
              optimizer=keras.optimizers.Adam(lr=0.005))

history = LossHistory()

x_,y_=dataset_maker(data_t)

x_train, x_val, y_train, y_val = train_test_split(x_, y_,train_size=0.8, random_state=33)

model.fit(x_train, y_train,
          batch_size=256, epochs=120,
          validation_data=(x_val, y_val),
          callbacks=[history])
pred_res=model.predict(x_val)*scaler.scale_[3]+scaler.mean_[3]
y_res=y_val*scaler.scale_[3]+scaler.mean_[3]
plot_model(model, show_shapes=True,rankdir='TB',to_file='model.png', dpi=100)

#history.loss_plot('epoch')
plt.figure(figsize=(4.5,2.5))
plt.rc('font',family='Times New Roman',size=10)
x_e=np.linspace(1,120,120)
plt.plot(x_e,history.val_loss['epoch'],c='r',lw=1.0)
plt.plot(x_e,history.losses['epoch'],c='dodgerblue',lw=1.0)
plt.ylim([0,0.2])
plt.xlim([0,120])
plt.grid()
plt.legend(['Val_loss','Train_loss'])
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.tight_layout()
plt.savefig('dnn_loss.png',dpi=300)
'''
plt.figure(figsize=(10,3))
x0=np.linspace(1,1800,1800)
plt.plot(x0,data.iloc[:1800,4])
'''
fig,ax=plt.subplots(3,2,figsize=(10,8))
plt.style.use('default')
#plt.rc('font',family='fangsong',size=6)
plt.rc('font',family='Times New Roman',size=10)
#fig.suptitle('沪深300指数走势预测')
#plt.title('沪深300指数走势预测')
xx=np.linspace(0.5,19.5,20)
ax[0][0].plot(xx,pred_res[60,:],c='r',ls='-.')
ax[0][0].plot(xx,y_res[60,:],c='dodgerblue',ls='-',marker='o',markersize=5,alpha=0.6)
font={'family':'fangsong','size':10}
ax[0][0].set_ylim([3200,3800])
ax[0][0].grid('on')
ax[0][0].set_xlabel('未来五个交易日（单位：小时）',font)
ax[0][0].set_xticks([0,5,10,15,20])
ax[0][0].set_xlim([0,20])
ax[0][0].set_ylabel('指数点位',font)
ax[0][0].legend(['DNN预测走势','实际走势'],loc='upper left',prop=font)

ax[0][1].plot(xx,pred_res[10,:],c='r',ls='-.')
ax[0][1].plot(xx,y_res[10,:],c='dodgerblue',ls='--',marker='o',markersize=5,alpha=0.6)
ax[0][1].set_ylim([3400,4000])
ax[0][1].grid('on')
font={'family':'fangsong','size':10}
ax[0][1].set_xlabel('未来五个交易日（单位：小时）',font)
ax[0][1].set_xticks([0,5,10,15,20])
ax[0][1].set_xlim([0,20])
ax[0][1].set_ylabel('指数点位',font)
ax[0][1].legend(['DNN预测走势','实际走势'],loc='upper left',prop=font)

ax[1][0].plot(xx,pred_res[82,:],c='r',ls='-.')
ax[1][0].plot(xx,y_res[82,:],c='dodgerblue',ls='-',marker='o',markersize=5,alpha=0.6)
ax[1][0].set_ylim([3100,3600])
ax[1][0].grid('on')
font={'family':'fangsong','size':10}
ax[1][0].set_xlabel('未来五个交易日（单位：小时）',font)
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlim([0,20])
ax[1][0].set_ylabel('指数点位',font)
ax[1][0].legend(['DNN预测走势','实际走势'],loc='upper left',prop=font)

ax[1][1].plot(xx,pred_res[45,:],c='r',ls='-.')
ax[1][1].plot(xx,y_res[45,:],c='dodgerblue',ls='-',marker='o',markersize=5,alpha=0.6)
ax[1][1].set_ylim([3600,4200])
ax[1][1].grid('on')
font={'family':'fangsong','size':10}
ax[1][1].set_xlabel('未来五个交易日（单位：小时）',font)
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlim([0,20])
ax[1][1].set_ylabel('指数点位',font)
ax[1][1].legend(['DNN预测走势','实际走势'],loc='upper left',prop=font)

ax[2][0].plot(xx,pred_res[30,:],c='r',ls='-.')
ax[2][0].plot(xx,y_res[30,:],c='dodgerblue',ls='-',marker='o',markersize=5,alpha=0.6)
ax[2][0].set_ylim([3600,4200])
ax[2][0].grid('on')
font={'family':'fangsong','size':10}
ax[2][0].set_xlabel('未来五个交易日（单位：小时）',font)
ax[2][0].set_xticks([0,5,10,15,20])
ax[2][0].set_xlim([0,20])
ax[2][0].set_ylabel('指数点位',font)
ax[2][0].legend(['DNN预测走势','实际走势'],loc='upper left',prop=font)

ax[2][1].plot(xx,pred_res[110,:],c='r',ls='-.')
ax[2][1].plot(xx,y_res[110,:],c='dodgerblue',ls='-',marker='o',markersize=5,alpha=0.6)
ax[2][1].set_ylim([3600,4200])
ax[2][1].grid('on')
font={'family':'fangsong','size':10}
ax[2][1].set_xlabel('未来五个交易日（单位：小时）',font)
ax[2][1].set_xticks([0,5,10,15,20])
ax[2][1].set_xlim([0,20])
ax[2][1].set_ylabel('指数点位',font)
ax[2][1].legend(['DNN预测走势','实际走势'],loc='upper left',prop=font)
plt.tight_layout()
plt.savefig('dnn_pred.png',dpi=300)