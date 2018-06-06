#-*-coding:utf-8-*-
#数据的预处理

import pandas as pd
import numpy as np

training_path = 'training.csv'
preliminary_path = 'preliminary-testing.csv'

#训练集替换
train_data = pd.read_csv(training_path, header=None, index_col=None)
# train_data = pd.DataFrame(train_data)

train_data1 = train_data.replace({'J':11,'Q':12,'K':13,'C':14,'D':15, 'H': 16,'S': 17})
train_data1.to_csv('new_train.csv', header=None, index=None)

#预测集替换
preliminary_data = pd.read_csv(preliminary_path, header=None, index_col=None)
# preliminary_data = pd.DataFrame(preliminary_data)

preliminary_data1 = preliminary_data.replace({'J':11,'Q':12,'K':13,'C':14,'D':15, 'H': 16,'S': 17})
preliminary_data1.to_csv('new_preliminary-testing.csv', header=None, index=None)