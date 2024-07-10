from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd 
import torch
import torch.utils.data as Data
from torch.autograd import Variable


# 文件读取
def get_Data(data_path):

    df = pd.read_excel(data_path)
    data = df.iloc[:,:]  # 数据
    label = df.iloc[:,-1] # 取最后一个特征作为标签
    print(data.head())
    print(label.head())
    return data,label


# 数据预处理
def normalization(data,label):

    mm_x = StandardScaler() # 导入sklearn的预处理容器
    mm_y = StandardScaler()
    data = data.values    # 将pd的系列格式转换为np的数组格式
    label = label
    data = mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    label = mm_y.fit_transform(label.values.reshape(-1, 1))
    return data,label,mm_y

# 时间向量转换
def split_windows(data,seq_length):

    x = []
    y = []
    for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _x = data[i:(i+seq_length),:]
        _y = data[i+seq_length,-1]
        x.append(_x)
        y.append(_y)
    x,y = np.array(x), np.array(y)
    print('x.shape, y.shape=\n',x.shape,y.shape)
    return x,y

# 数据分离
def split_data(x,y,split_ratio):

    train_size = int(len(y)*split_ratio)
    test_size = len(y)-train_size

    x_data = Variable(torch.Tensor(np.array(x)))
    y_data = Variable(torch.Tensor(np.array(y)))

    x_train = Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train = Variable(torch.Tensor(np.array(y[0:train_size])))
    y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    x_test = Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
    .format(x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape))

    return x_data,y_data,x_train,y_train,x_test,y_test

# 数据装入
def data_generator(x_data,y_data,x_train,y_train,x_test,y_test,batch_size):

    train_dataset = Data.TensorDataset(x_data,y_data)  ##########
    test_dataset = Data.TensorDataset(x_test,y_test)  ############
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=True) # 加载数据集,使数据集可迭代
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

    return train_loader,test_loader