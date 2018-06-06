#-*-coding:utf-8-*-


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pylab import *

def data_change():
    raw_pd = pd.read_csv('training.csv')
    raw_x = raw_pd.drop(['9'], axis=1)
    raw_x_drop = raw_x.drop_duplicates()
    raw_pd = raw_pd.replace({'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13})
    raw_drop = raw_pd.drop_duplicates()
    m = np.shape(raw_pd)[0]
    n = np.shape(raw_drop)[0]
    count = m - n
    print(m, n)
    return raw_pd, count


#将纸牌的大小排序，求证label跟纸牌的顺序有关
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
    x_data = pd.DataFrame(x_data)
    print(x_data)
    print('排序后未加label重复数据条数：')
    data_drop1 = x_data.drop_duplicates()
    print(np.shape(x_data)[0] - np.shape(data_drop1)[0])
    print('排序后合并label重复数据条数：')
    data_drop2 = x_data.drop_duplicates()
    print(np.shape(data)[0] - np.shape(data_drop2)[0])
    x_data.columns = ['a','b','c','d','e','f','g','h','i','j','k']
    return x_data

#统计纸牌之和和分类的关系
def data_sum(data):
    sum = data['10'] + data['J'] + data['Q'] + data['K'] + data['1']
    raw_y = data['9']
    sum = pd.DataFrame(sum)
    
    # print(data_sum.sort_values(['9']))
    
    #花色统计
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

    return data_t
    
    
#纸牌和纸牌数目之和的关系
def tiaoxingtu(sum):
    sum = data['10'] + data['J'] + data['Q'] + data['K'] + data['1']
    raw_y = data['9']
    sum = pd.DataFrame(sum)
    data_count = sum.join(raw_y)
    
    x = []
    y = []
    x_0 = [];x_1 = [];x_2 = [];x_3 = [];x_4 = [];x_5 = [];x_6 = [];x_7 = [];x_8 = [];x_9 = []
    
    for i, row in enumerate(data_count.values):
        # index = data_count[i]
        x1 = row[0]
        y1 = row[1]
        x.append(x1)
        y.append(y1)
        if y1 == 0:
            x_0.append(x1)
        elif y1 == 1:
            x_1.append(x1)
        elif y1 == 2:
            x_2.append(x1)
        elif y1 == 3:
            x_3.append(x1)
        elif y1 == 4:
            x_4.append(x1)
        elif y1 == 5:
            x_5.append(x1)
        elif y1 == 6:
            x_6.append(x1)
        elif y1 == 7:
            x_7.append(x1)
        elif y1 == 8:
            x_8.append(x1)
        else:
            x_9.append(x1)
    
    #画出条形直方图显示sum和纸牌的关系
    x_label = range(len(x_5))
    rects1 = plt.bar(left=x_label, height=x_5, width=0.4, alpha=0.8, color='red', label="一部门")
    # rects2 = plt.bar(left=[i + 0.4 for i in x], height=num_list2, width=0.4, color='green', label="二部门")
    plt.ylim(0, 53)  # y轴取值范围
    plt.ylabel("数量")
    plt.legend()  # 设置题注
    
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2 , height + 1, str(height), ha="center", va="bottom")
    # for rect in rects2:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.show()
    
    
    #统计每个纸牌的样本个数
    data_values = dict()
    data_values['0'] = y.count(0)
    data_values['1'] = y.count(1)
    data_values['2'] = y.count(2)
    data_values['3'] = y.count(3)
    data_values['4'] = y.count(4)
    data_values['5'] = y.count(5)
    data_values['6'] = y.count(6)
    data_values['7'] = y.count(7)
    data_values['8'] = y.count(8)
    data_values['9'] = y.count(9)
    print('各个样本个数如下：')
    print(data_values)
  
    
    
    
if __name__=='__main__':
    data, count = data_change()
    
    x = data_sort(data)
    print(x)
    # t = data_sum(data)
    # tiaoxingtu(data)
    # data_sort(data)
    print('数据重复 %d条'% count)
    
    
'''
个样本的统计
{'0': 12493, '1': 513, '2': 1206, '3': 10599, '4': 93, '5': 54, '6': 36, '7': 6, '8': 5, '9': 4}
排序前数据中2条重复
排序后合并label前后重复数据条数相同83条
不同的组合对应不同的纸牌
提取sum和花色统计特征
'''