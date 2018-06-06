#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import neighbors


def data():
    train_path = 'training.csv'
    preliminary_path = 'new_preliminary-testing.csv'
    
    #训练集
    data = pd.read_csv(train_path, header=None, index_col=None)
    data1 = data.sample(frac= 1) # 打乱样本
    data1 = pd.DataFrame(data1)
    data_replace = data1.replace({'J':11,'Q':12,'K':13,'C':0.52,'D':0.54,'S':0.56,'H':0.58})
    # data1.columns = ['A', 'A1', 'B', 'B1', 'C', 'C1', 'D', 'D1', 'E', 'E1', 'L']

    #预测集
    
    data2 = pd.read_csv(preliminary_path, header=None, index_col=None)
    
    
    #分割样本--留出法
    X_train, X_test, Y_train, Y_test = train_test_split(
        data_replace.loc[:,:9], data_replace.loc[:,10], test_size=0.30, random_state=42)
    print(data_replace.loc[:,:9])
    #将纸牌的点数和纸牌的花色做加法训练数据

    return X_train, X_test, Y_train, Y_test, data1

def data_dealwith(train, test):
    #数据归一化
    train_data = preprocessing.StandardScaler().fit(train)
    test_data = preprocessing.StandardScaler().fit(test)
    return train_data.transform(train), test_data.transform(test)



#KNN
def knn_model(X_train, X_test, Y_train, Y_test):
    # knn
    model = neighbors.KNeighborsClassifier(n_neighbors=10, n_jobs=1)
    # score = cross_val_score(model, data1.loc[:,:9],data1.loc[:,9],cv=5)  #交叉验证
    model.fit(X_train, Y_train)
    l = model.predict(X_test)
    s1 = model.get_params()
    ss = model.score(X_test, Y_test)  # 预测正确率
    return l, ss


if __name__=='__main__':
    
    X_train, X_test, Y_train, Y_test, data1 = data()
    x_train_g, x_test_g = data_dealwith(X_train, X_test)  #效果较差
    #knn模型
    l, ss = knn_model(x_train_g, x_test_g, Y_train, Y_test)







#显示结果========================================================
    s = 0
    t = 0
    for i in Y_test:
        print(i, '-->', l[s])
        if i == l[s]:
            t += 1
        s += 1
    print(s, t)
    print('+++++++++++++++++++++++++++')
    print(s)
    print('===========================')
    print(ss)
    # trees(train, test)
