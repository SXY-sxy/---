import pandas as pd
import numpy as np
from itertools import permutations

raw_data = pd.read_csv('training.csv', header=None)
raw_data.head()

for i in range(0,10,2):
    raw_data['poker_'+str(i//2+1)] = raw_data[i] + '_' + raw_data[i+1]
    raw_data = raw_data.drop([i,i+1],axis=1)
label = raw_data.iloc[:,0]
raw_data = raw_data.drop([10], axis=1)
raw_data['label'] = label

'''
#按照label排序，增加label为789的样本

'''
raw_data2 = raw_data.sort_values(by='label',ascending = True)
print(raw_data2)


result = raw_data.copy()
for index in permutations(range(0,5),5):
    
    index = list(index)
    index.append(5)
    tmp = raw_data.iloc[24910:,index]
    tmp.columns = ['poker_1','poker_2','poker_3','poker_4','poker_5','label']
    result = pd.concat([result, tmp])
result.drop_duplicates(inplace=True)

for col in ['poker_1','poker_2','poker_3','poker_4','poker_5']:
    
    col_data = result[col].tolist()
    result = result.drop([col], axis=1)
    col_data = [item.split('_') for item in col_data]
    col_data = np.array(col_data)
    result[col + '_color'] = col_data[:,0]
    result[col + '_size'] = col_data[:,1]
    
    
result.to_csv('training3.csv', index=None)