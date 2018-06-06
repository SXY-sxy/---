import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import zhipai.net_model
from zhipai.data_fenxi import data_sort


def feature_extraction():
    raw_pd = pd.read_csv('training.csv')
   
    #不排序的数据
    # raw_pd1 = raw_pd.sample(frac=1)
    # raw_x = raw_pd1.drop(['9'], axis=1)
    # raw_y = raw_pd1['9']
    # raw_y = np.ravel(raw_y)
    
    # raw_x = raw_pd.drop(['k'], axis=1)


    # 将数据属性排序
    raw_x = raw_pd.drop(['9'], axis=1)
    raw_y = raw_pd['9']
    x_data = []

    for i, row in enumerate(raw_x.values):
        x_label = [row[0], row[2], row[4], row[6], row[8]]
        x_value = [row[1], row[3], row[5], row[7], row[9]]

        # for i in range(len(array) - 1):
        #     for j in range(len(array) - i - 1):
        #         if array[j] > array[j + 1]:
        #             array[j], array[j + 1] = array[j + 1], array[j]
        for i in range(len(x_value)-1):
            for j in range(i, len(x_value)-i-1):
                if x_value[j] > x_value[j+1]:
                    x_value[j], x_value[j+1] = x_value[j+1], x_value[j]
                    x_label[j], x_label[j+1] = x_label[j+1], x_label[j]
        x_d = [x_label[0], x_value[0], x_label[1], x_value[1], x_label[2], x_value[2], x_label[3], x_value[3],
               x_label[4], x_value[4]]
        x_data.append(x_d)
    x_data = pd.DataFrame(x_data)
    print(x_data)
    print('排序后未加label重复数据条数：')
    data_drop1 = x_data.drop_duplicates()
    print(np.shape(x_data)[0] - np.shape(data_drop1)[0])
    print('排序后合并label重复数据条数：')
    x_data_mer = x_data.join(raw_y)
    data_drop2 = x_data_mer.drop_duplicates()
    print(np.shape(raw_pd)[0] - np.shape(data_drop2)[0])
    x_data_mer1 = pd.DataFrame(x_data_mer)
    x_data_mer1.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

    
    raw_pd1 = x_data_mer.sample(frac=1)
    
    print(raw_pd1)
    print('==========================')
    raw_y = raw_pd1['k']
    raw_y = np.ravel(raw_y)
    raw_x = raw_pd1.drop(['k'], axis=1)
    
    
    # 对所有特征进行one-hot编码
    raw_x = pd.get_dummies(raw_x)
    raw_y = pd.get_dummies(raw_y)
    print(np.shape(raw_x))
    pca = PCA(n_components='mle',svd_solver='full', whiten=False)
    pca.fit(raw_x)
    print(pca.explained_variance_)
    print('===================================')
    print(pca.explained_variance_ratio_)
    print(pca.n_components_)
    print(pd.DataFrame(pca.fit_transform(raw_x)))
    
    
if __name__=='__main__':
    feature_extraction()