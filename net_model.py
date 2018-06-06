#-*-coding:utf-8-*-

'''
防止提交结果错位，手动将预测集的第一列复制到最后一列
数据量100001
'''
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,Callback
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
from matplotlib.pyplot import plot as plt, figure
from pylab import *


#读取数据
def load_data():
    raw_pd = pd.read_csv('training3.csv')
    '''
        #将数据属性排序
        raw_x = data_sort(raw_pd)

        raw_y = raw_pd['9']
        raw_y = np.ravel(raw_y)

        data = raw_x.join(raw_y)
        data = data.drop_duplicates()
        raw_x = data.drop(['k'], axis=1)
        raw_y = raw_pd['9']
        raw_y = np.ravel(raw_y)
        '''
    raw_pd1 = raw_pd.sample(frac=1)
    raw_x = raw_pd1.drop(['label'], axis=1)
    raw_y = raw_pd1['label']
    raw_y = np.ravel(raw_y)
    
    # raw_x = raw_pd.drop(['k'], axis=1)
    
    # 对所有特征进行one-hot编码
    raw_x = pd.get_dummies(raw_x)
    raw_y = pd.get_dummies(raw_y)
    
    #添加纸牌和，花色统计特征
    data = raw_pd1.replace({'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13})
    sum = data['poker_1_size'] + data['poker_2_size'] + data['poker_3_size'] + data['poker_4_size'] + data['poker_5_size']
    # sum = data['b'] + data['d'] + data['f'] + data['h'] + data['j']
    sum = pd.DataFrame(sum)

 
    # 花色统计
    data_huase = []
    for i in range(len(data)):
        data_c = data_s = data_d = data_h = 0
    
        # if data.loc[i]['H'] == 'C':
        if data.loc[i]['poker_1_color'] == 'C':
            data_c += 1
        elif data.loc[i]['poker_1_color'] == 'S':
            data_s += 1
        elif data.loc[i]['poker_1_color'] == 'D':
            data_d += 1
        elif data.loc[i]['poker_1_color'] == 'H':
            data_h += 1
    
        if data.loc[i]['poker_2_color'] == 'C':
            data_c += 1
        elif data.loc[i]['poker_2_color'] == 'S':
            data_s += 1
        elif data.loc[i]['poker_2_color'] == 'D':
            data_d += 1
        elif data.loc[i]['poker_2_color'] == 'H':
            data_h += 1
    
        if data.loc[i]['poker_3_color'] == 'C':
            data_c += 1
        elif data.loc[i]['poker_3_color'] == 'S':
            data_s += 1
        elif data.loc[i]['poker_3_color'] == 'D':
            data_d += 1
        elif data.loc[i]['poker_3_color'] == 'H':
            data_h += 1
    
        if data.loc[i]['poker_4_color'] == 'C':
            data_c += 1
        elif data.loc[i]['poker_4_color'] == 'S':
            data_s += 1
        elif data.loc[i]['poker_4_color'] == 'D':
            data_d += 1
        elif data.loc[i]['poker_4_color'] == 'H':
            data_h += 1
    
        if data.loc[i]['poker_5_color'] == 'C':
            data_c += 1
        elif data.loc[i]['poker_5_color'] == 'S':
            data_s += 1
        elif data.loc[i]['poker_5_color'] == 'D':
            data_d += 1
        elif data.loc[i]['poker_5_color'] == 'H':
            data_h += 1
        data_huase.append([data_c, data_d, data_s, data_h])
        # print('样本纸牌花色的个数:')
        # print('data_c: %d' % data_c)
        # print('data_s: %d' % data_s)
        # print('data_d: %d' % data_d)
        # print('data_h: %d' % data_h)
    data_huase = pd.DataFrame(data_huase, columns=['c', 'd', 's', 'h'])
    data_t = sum.join(data_huase)
    raw_x = raw_x.join(data_t)
    # raw_x = raw_x.join(data.drop(['9'], axis=1))
    print(raw_x)
    print(np.shape(raw_x))
    
    X_train, X_test, Y_train, Y_test = train_test_split(raw_x, raw_y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test


#预测将所有数据训练
def load_data2():
    raw_pd = pd.read_csv('training.csv')
    raw_y = raw_pd['9']
    raw_y = np.ravel(raw_y)
    raw_x = raw_pd.drop(['9'], axis=1)
    # 对所有特征进行one-hot编码
    raw_x = pd.get_dummies(raw_x)
    raw_y = pd.get_dummies(raw_y)

    # 添加纸牌和，花色统计特征
    data = raw_pd.replace(
        {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13})
    sum = data['10'] + data['J'] + data['Q'] + data['K'] + data['1']
    # sum = data['b'] + data['d'] + data['f'] + data['h'] + data['j']
    sum = pd.DataFrame(sum)

    # 花色统计
    data_huase = []
    for i in range(len(data)):
        data_c = data_s = data_d = data_h = 0
    
        if data.loc[i]['H'] == 'C':
            data_c += 1
        elif data.loc[i]['H'] == 'S':
            data_s += 1
        elif data.loc[i]['H'] == 'D':
            data_d += 1
        elif data.loc[i]['H'] == 'H':
            data_h += 1
    
        if data.loc[i]['H.1'] == 'C':
            data_c += 1
        elif data.loc[i]['H.1'] == 'S':
            data_s += 1
        elif data.loc[i]['H.1'] == 'D':
            data_d += 1
        elif data.loc[i]['H.1'] == 'H':
            data_h += 1
    
        if data.loc[i]['H.2'] == 'C':
            data_c += 1
        elif data.loc[i]['H.2'] == 'S':
            data_s += 1
        elif data.loc[i]['H.2'] == 'D':
            data_d += 1
        elif data.loc[i]['H.2'] == 'H':
            data_h += 1
    
        if data.loc[i]['H.3'] == 'C':
            data_c += 1
        elif data.loc[i]['H.3'] == 'S':
            data_s += 1
        elif data.loc[i]['H.3'] == 'D':
            data_d += 1
        elif data.loc[i]['H.3'] == 'H':
            data_h += 1
    
        if data.loc[i]['H.4'] == 'C':
            data_c += 1
        elif data.loc[i]['H.4'] == 'S':
            data_s += 1
        elif data.loc[i]['H.4'] == 'D':
            data_d += 1
        elif data.loc[i]['H.4'] == 'H':
            data_h += 1
        data_huase.append([data_c, data_d, data_s, data_h])
        # print('样本纸牌花色的个数:')
        # print('data_c: %d' % data_c)
        # print('data_s: %d' % data_s)
        # print('data_d: %d' % data_d)
        # print('data_h: %d' % data_h)
    data_huase = pd.DataFrame(data_huase, columns=['c', 'd', 's', 'h'])
    data_t = sum.join(data_huase)
    raw_x = raw_x.join(data_t)
    # raw_x = raw_x.join(data.drop(['9'], axis=1))
    print(raw_x)
    print(np.shape(raw_x))
    return raw_x, raw_y


#神经网络
#获取保存的效果最好的loss
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
        
def net_model(X_train, X_test, Y_train, Y_test):
    model = Sequential()
    model.add(Dense(128, input_shape=(90,)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim=128, output_dim=256))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(input_dim=256, output_dim=128))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.2))
    # model.add(Dense(input_dim=512, output_dim=128))
    # model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=128, output_dim=10))
    model.add(Activation('softmax'))
    
    #重新设置学习率
    adams = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0.0, amsgrad=False)
    # model.compile(optimizer=adams, loss='mean_squared_error', metrics=["accuracy"])#loss:均方误差
    model.compile(optimizer=adams, loss='categorical_crossentropy', metrics=["accuracy"])  # loss:均方误差
    
    
    # 用于保存验证集误差最小的参数，当验证集误差减少时，立马保存下来
    checkpointer = ModelCheckpoint(filepath="mouth.hdf5", verbose=1, save_best_only=True)
    history = LossHistory()
    
    
    hist = model.fit(X_train, Y_train, nb_epoch=800, batch_size=100, validation_split=0.1, callbacks=[checkpointer, history])#validation_split：交差验证

    '''
    pre = model.predict(X_test)
    t = 0
    for i in pre:
        t += 1
        dighit = np.argmax(i)
        print('第%d次:'%t, end='\t')
        print(dighit)
        print('=====================================')
    '''
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
    
    return loss, accuracy, model, hist
    

#loss+val_loss, acc+val_acc变化图
def plot_loss_acc(hist):
    loss = hist.history['loss']
    acc = hist.history['acc']
    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
    
def predict_model(model):
    raw_pd = pd.read_csv('preliminary-testing.csv')
    print(raw_pd)
    raw_x = pd.get_dummies(raw_pd)
    raw_x = pd.DataFrame(raw_x)
    
    # 添加纸牌和，花色统计特征
    data = raw_pd.replace(
        {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13})
    sum = data['5'] + data['7'] + data['9'] + data['3'] + data['K']
    sum = pd.DataFrame(sum)

    # 花色统计
    data_huase = []
    for i in range(len(data)):
        data_c = data_s = data_d = data_h = 0
    
        if data.loc[i]['C'] == 'C':
            data_c += 1
        elif data.loc[i]['C'] == 'S':
            data_s += 1
        elif data.loc[i]['C'] == 'D':
            data_d += 1
        elif data.loc[i]['C'] == 'H':
            data_h += 1
    
        if data.loc[i]['S'] == 'C':
            data_c += 1
        elif data.loc[i]['S'] == 'S':
            data_s += 1
        elif data.loc[i]['S'] == 'D':
            data_d += 1
        elif data.loc[i]['S'] == 'H':
            data_h += 1
    
        if data.loc[i]['H'] == 'C':
            data_c += 1
        elif data.loc[i]['H'] == 'S':
            data_s += 1
        elif data.loc[i]['H'] == 'D':
            data_d += 1
        elif data.loc[i]['H'] == 'H':
            data_h += 1
    
        if data.loc[i]['D'] == 'C':
            data_c += 1
        elif data.loc[i]['D'] == 'S':
            data_s += 1
        elif data.loc[i]['D'] == 'D':
            data_d += 1
        elif data.loc[i]['D'] == 'H':
            data_h += 1
    
        if data.loc[i]['C.1'] == 'C':
            data_c += 1
        elif data.loc[i]['C.1'] == 'S':
            data_s += 1
        elif data.loc[i]['C.1'] == 'D':
            data_d += 1
        elif data.loc[i]['C.1'] == 'H':
            data_h += 1
        data_huase.append([data_c, data_d, data_s, data_h])
        # print('样本纸牌花色的个数:')
        # print('data_c: %d' % data_c)
        # print('data_s: %d' % data_s)
        # print('data_d: %d' % data_d)
        # print('data_h: %d' % data_h)
    data_huase = pd.DataFrame(data_huase, columns=['c', 'd', 's', 'h'])
    data_t = sum.join(data_huase)
    raw_pre = raw_x.join(data_t)
    results = model.predict(raw_pre)
    
    
    f = open('1.txt', 'w')
    for i in results:
        dighit = np.argmax(i)
        f.write(str(dighit)+'\n')
    f.close()


'''
def data_sort(data):
    raw_x = data.drop(['9'], axis=1)
    raw_y = data['9']
    x_data = []
    
    for i, row in enumerate(raw_x.values):
        x_label = [row[0], row[2], row[4], row[6], row[8]]
        x_value = [row[1], row[3], row[5], row[7], row[9]]
        
        for i in range(len(x_value)):
            for j in range(i, len(x_value)):
                if x_value[i] > x_value[j]:
                    x_value[i], x_value[j] = x_value[j], x_value[i]
                    x_label[i], x_label[j] = x_label[j], x_label[i]
        x_d = [x_label[0], x_value[0], x_label[1], x_value[1], x_label[2], x_value[2], x_label[3], x_value[3],
               x_label[4], x_value[4]]
        x_data.append(x_d)
    x_data = pd.DataFrame(x_data, columns=['a','b','c','d','e','f','g','h','i','j'])
    return x_data
'''


if __name__=='__main__':
   
    # X_train, X_test, Y_train, Y_test = load_data()
    #读取数据
    X_train, X_test, Y_train, Y_test = load_data()
    raw_x, raw_y = load_data2()
    
    #训练
    loss, accuracy, model, hist = net_model(X_train, X_test, Y_train, Y_test)
    #将全部数据训练
    # loss, accuracy, model, hist = net_model(raw_x, X_test, raw_y, Y_test)
    plot_loss_acc(hist)
    #预测
    # predict_model(model)
    print(loss, accuracy)
 