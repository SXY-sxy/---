#-*-coding:utf-8-*-

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def load_data():
    filename = 'training.csv'
    data_pd = pd.read_csv(filename, header=None)
    # data_pd = data_p.sample(frac=1)
    data_1 = data_pd.replace({'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':11, 'Q':12, 'K':13})
    data_sor = data_sort(data_1)
    # data_sor1 = data_sor.drop_duplicates()
    data_2 = data_sor.replace({1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10', 11:'J', 12:'Q', 13:'K'})
    huase = huase_count(data_sor)
    data_3 = data_pd.drop([10], axis=1)
    
    
    #one-hot编码
    raw_x = pd.get_dummies(data_3)
    print(np.shape(raw_x))
    print('=====================')
    Y_train = data_pd[10]
    x_train = pd.DataFrame()
    x_train['sub_1'] = data_sor['b'] - 0
    x_train['sub_2'] = data_sor['d'] - data_sor['b']
    x_train['sub_3'] = data_sor['f'] - data_sor['d']
    x_train['sub_4'] = data_sor['h'] - data_sor['f']
    x_train['sub_5'] = data_sor['j'] - data_sor['h']
    x_train = x_train.join(raw_x)
    x_train['hua_c'] = huase['hua_c']
    x_train['hua_d'] = huase['hua_d']
    x_train['hua_s'] = huase['hua_s']
    x_train['hua_h'] = huase['hua_h']
    
    x_train['sum'] = data_sor['b'] + data_sor['d'] + data_sor['f'] + data_sor['h'] + data_sor['j']
    x_train['mau'] = data_sor['b'] * data_sor['d'] * data_sor['f'] * data_sor['h'] * data_sor['j']
    x_train['avg'] = x_train['sum'] / 5
    x_train['std'] = ((data_sor['b']-x_train['avg'])**2 + (data_sor['d'] - x_train['avg'])**2 + (data_sor['f'] - x_train['avg'])**2 +\
                     (data_sor['h'] - x_train['avg'])**2 + (data_sor['j'] - x_train['avg'])**2) / 5
    
    
    contiunous = []
    for i in range(len(x_train)):
        if x_train.loc[i]['sub_2'] == x_train.loc[i]['sub_3'] == x_train.loc[i]['sub_4'] == x_train.loc[i]['sub_5'] == 1:
            contiunous.append(1)
            print(i)
        else:
            contiunous.append(0)
    contiunous = pd.DataFrame(contiunous, columns=['conti'])
    X_train = x_train.join(contiunous)
            
    print(X_train)
    
    
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2)
    return x_train, x_test, y_train, y_test, X_train, Y_train


#统计纸牌各个花色的个数
def huase_count(data):
    huase = []
    for i in range(len(data)):
        hua_c = hua_d = hua_s = hua_h = 0
        if data.loc[i]['a'] == 'C':
            hua_c += 1
        elif data.loc[i]['a'] == 'D':
            hua_d += 1
        elif data.loc[i]['a'] == 'S':
            hua_s += 1
        else:
            hua_h += 1
            
        if data.loc[i]['c'] == 'C':
            hua_c += 1
        elif data.loc[i]['c'] == 'D':
            hua_d += 1
        elif data.loc[i]['c'] == 'S':
            hua_s += 1
        else:
            hua_h += 1
            
        if data.loc[i]['e'] == 'C':
            hua_c += 1
        elif data.loc[i]['e'] == 'D':
            hua_d += 1
        elif data.loc[i]['e'] == 'S':
            hua_s += 1
        else:
            hua_h += 1
            
        if data.loc[i]['g'] == 'C':
            hua_c += 1
        elif data.loc[i]['g'] == 'D':
            hua_d += 1
        elif data.loc[i]['g'] == 'S':
            hua_s += 1
        else:
            hua_h += 1
            
        if data.loc[i]['i'] == 'C':
            hua_c += 1
        elif data.loc[i]['i'] == 'D':
            hua_d += 1
        elif data.loc[i]['i'] == 'S':
            hua_s += 1
        else:
            hua_h += 1
        huase.append([hua_c, hua_d, hua_s, hua_h])
    huase = pd.DataFrame(huase, columns=['hua_c', 'hua_d', 'hua_s', 'hua_h'])
    return huase
    
    
#纸牌按从小到大排序
def data_sort(data):
    x_data = []
    for i, row in enumerate(data.values):
        x_label = [row[0],row[2],row[4],row[6],row[8]]
        x_value = [row[1],row[3],row[5],row[7],row[9]]

        for i in range(len(x_value)-1):
            for j in  range(i, len(x_value)):
                if x_value[i] > x_value[j]:
                    x_value[i], x_value[j] = x_value[j], x_value[i]
                    x_label[i], x_label[j] = x_label[j], x_label[i]
        x_d = [x_label[0],x_value[0],x_label[1],x_value[1],x_label[2],x_value[2],x_label[3],x_value[3],x_label[4],x_value[4], row[10]]
        x_data.append(x_d)
    print('排序后未加label重复数据条数：')
    x_data = pd.DataFrame(x_data)
    data_drop1 = x_data.drop_duplicates()
    print(np.shape(x_data)[0] - np.shape(data_drop1)[0])
    print('排序后合并label重复数据条数：')
    data_drop2 = x_data.drop_duplicates()
    print(np.shape(data)[0] - np.shape(data_drop2)[0])
    x_data.columns = ['a','b','c','d','e','f','g','h','i','j','k']
    return x_data

#纸牌按从小到大排序
def data_sortpre(data):
    x_data = []
    for i, row in enumerate(data.values):
        x_label = [row[0],row[2],row[4],row[6],row[8]]
        x_value = [row[1],row[3],row[5],row[7],row[9]]

        for i in range(len(x_value)-1):
            for j in  range(i, len(x_value)):
                if x_value[i] > x_value[j]:
                    x_value[i], x_value[j] = x_value[j], x_value[i]
                    x_label[i], x_label[j] = x_label[j], x_label[i]
        x_d = [x_label[0],x_value[0],x_label[1],x_value[1],x_label[2],x_value[2],x_label[3],x_value[3],x_label[4],x_value[4]]
        x_data.append(x_d)
    print('排序后未加label重复数据条数：')
    x_data = pd.DataFrame(x_data)
    data_drop1 = x_data.drop_duplicates()
    print(np.shape(x_data)[0] - np.shape(data_drop1)[0])
    print('排序后合并label重复数据条数：')
    data_drop2 = x_data.drop_duplicates()
    print(np.shape(data)[0] - np.shape(data_drop2)[0])
    x_data.columns = ['a','b','c','d','e','f','g','h','i','j']
    return x_data

#模型
def trees(x_train, x_test, y_train, y_test):
    #决策树
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    score_test = clf.score(x_test, y_test)
    print(score_test)
    score = cross_val_score(clf, x_train, y_train, cv=10)
    print(score)
    
    #随机森林
    # clf = RandomForestClassifier()
    # clf.fit(x_train, y_train)
    # score_test = clf.score(x_test, y_test)
    # print(score_test)
    # score = cross_val_score(clf, x_train, y_train, cv=10)
    # score_t = clf.score(x_test, y_test)
    # print(score)
    # print(score_t)
    return clf


def predict(clf):
    filename = 'preliminary-testing.csv'
    data_pd= pd.read_csv(filename, header=None)
    # data_pd = data_p.sample(frac=1)
    data_1 = data_pd.replace(
        {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13})
    data_sor = data_sortpre(data_1)
    # data_sor1 = data_sor.drop_duplicates()
    data_2 = data_sor.replace(
        {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'})
    huase = huase_count(data_sor)

    # one-hot编码
    raw_x = pd.get_dummies(data_pd)
    print(np.shape(raw_x))
    print('*************************')
    
    
    x_train = pd.DataFrame()
    x_train['sub_1'] = data_sor['b'] - 0
    x_train['sub_2'] = data_sor['d'] - data_sor['b']
    x_train['sub_3'] = data_sor['f'] - data_sor['d']
    x_train['sub_4'] = data_sor['h'] - data_sor['f']
    x_train['sub_5'] = data_sor['j'] - data_sor['h']
    x_train = x_train.join(raw_x)
    x_train['hua_c'] = huase['hua_c']
    x_train['hua_d'] = huase['hua_d']
    x_train['hua_s'] = huase['hua_s']
    x_train['hua_h'] = huase['hua_h']
    
    x_train['sum'] = data_sor['b'] + data_sor['d'] + data_sor['f'] + data_sor['h'] + data_sor['j']
    x_train['mau'] = data_sor['b'] * data_sor['d'] * data_sor['f'] * data_sor['h'] * data_sor['j']
    x_train['avg'] = x_train['sum'] / 5
    x_train['std'] = ((data_sor['b'] - x_train['avg']) ** 2 + (data_sor['d'] - x_train['avg']) ** 2 + (
    data_sor['f'] - x_train['avg']) ** 2 + \
                      (data_sor['h'] - x_train['avg']) ** 2 + (data_sor['j'] - x_train['avg']) ** 2) / 5
    
    contiunous = []
    for i in range(len(x_train)):
        if x_train.loc[i]['sub_2'] == x_train.loc[i]['sub_3'] == x_train.loc[i]['sub_4'] == x_train.loc[i][
            'sub_5'] == 1:
            contiunous.append(1)
            print(i)
        else:
            contiunous.append(0)
    contiunous = pd.DataFrame(contiunous, columns=['conti'])
    pre_train = x_train.join(contiunous)
    
    results = clf.predict(pre_train)
    f = open('1.txt', 'w')
    for i in results:
        f.write(str(i) + '\n')
    f.close()
    print(i)
    
    print('结束！！')
    
if __name__=='__main__':
    x_train, x_test, y_train, y_test, prex_train, prey_train = load_data()
    clf = trees(prex_train, x_test, prey_train, y_test)
    predict(clf)