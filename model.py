from tcn import *
from fightingcv_attention.attention.ExternalAttention import ExternalAttention 


class AN(nn.Module):
    def __init__(self,num_inputs) -> None:
        super(AN, self).__init__()
        self.TCN = TemporalConvNet(num_inputs, num_channels=[48], kernel_size=3, dropout=0.5)   # 48, 3, 0.5
        self.SA =  ExternalAttention(48, 24)                                                    # 48, 24
        self.relu1 = nn.ReLU(inplace=False)
        self.RNN = nn.LSTM(48, 24, 1)                                               # 48, 24, 1,0.3 LSTM
        self.relu2 = nn.ReLU(inplace=False)
        self.output_f = nn.Linear(24, 1)                                                        # 24, 1
        # self.relu3 = nn.ReLU(inplace=True)
        # self.output_l = nn.Linear(20, 1)
        # self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):     # 32, 20, 12  (Bath size, length, channel)
        # print(f'输入:{x.shape}')
        x = x.permute(0,2,1)  # 32, 12, 20
        # print(f'交换:{x.shape}')
        x = self.TCN(x)   
        # print(f'TCN后:{x.shape}')
        x = x.permute(0,2,1)   
        # print(f'交换:{x.shape}')
        x = self.SA(x)
        # print(f'SA后:{x.shape}')
        x = self.relu1(x)
        x, _ = self.RNN(x)
        # print(f'RNN后:{x.shape}')
        x = self.relu2(x)
        x = self.output_f(x)

        return x[:,-1,:].squeeze(-1)
    


class DO(nn.Module):
    def __init__(self,num_inputs) -> None:
        super(DO, self).__init__()
        self.TCN = TemporalConvNet(num_inputs, num_channels=[24, 48], kernel_size=3, dropout=0.5)   # [24,48], 3, 0.5
        self.SA =  ExternalAttention(48, 24)                                                    # 48, 24
        self.relu1 = nn.ReLU(inplace=False)
        self.RNN = nn.LSTM(48, 24, 2,dropout = 0.3)                                               # 48, 24, 2,0.3 LSTM
        self.relu2 = nn.ReLU(inplace=False)
        self.output_f = nn.Linear(24, 1)                                                        # 24, 1
        # self.relu3 = nn.ReLU(inplace=True)
        # self.output_l = nn.Linear(20, 1)
        # self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):     # 32, 20, 12  (Bath size, channel, length)
        # print(f'输入:{x.shape}')
        x = x.permute(0,2,1)  # 32, 12, 20
        # print(f'交换:{x.shape}')
        x = self.TCN(x)   
        # print(f'TCN后:{x.shape}')
        x = x.permute(0,2,1)   
        # print(f'交换:{x.shape}')
        x = self.SA(x)
        # print(f'SA后:{x.shape}')
        x = self.relu1(x)
        x, _ = self.RNN(x)
        # print(f'RNN后:{x.shape}')
        x = self.relu2(x)
        x = self.output_f(x)
        # print(f'Feature输出:{x.shape}')
        # x = self.relu3(x)
        # x = x.permute(0,2,1)
        # x = self.output_l(x)
        # x = x.permute(0,2,1)
        # print(f'len输出:{x.shape}')
        # return x.squeeze(-1)
        return x[:,-1,:].squeeze(-1)

class PH(nn.Module):
    def __init__(self,num_inputs) -> None:
        super(PH, self).__init__()
        self.TCN = TemporalConvNet(num_inputs, num_channels=[24, 48], kernel_size=3, dropout=0.5)   # 48, 3, 0.5
        self.SA =  ExternalAttention(48, 24)                                                    # 48, 24
        self.relu1 = nn.ReLU(inplace=False)
        self.RNN = nn.LSTM(48, 24, 1)                                               # 48, 24, 1,0.3 LSTM
        self.relu2 = nn.ReLU(inplace=False)
        self.output_f = nn.Linear(24, 1)                                                        # 24, 1
        # self.relu3 = nn.ReLU(inplace=True)
        # self.output_l = nn.Linear(20, 1)
        # self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):     # 32, 20, 12  (Bath size, channel, length)
        # print(f'输入:{x.shape}')
        x = x.permute(0,2,1)  # 32, 12, 20
        # print(f'交换:{x.shape}')
        x = self.TCN(x)   
        # print(f'TCN后:{x.shape}')
        x = x.permute(0,2,1)   
        # print(f'交换:{x.shape}')
        x = self.SA(x)
        # print(f'SA后:{x.shape}')
        x = self.relu1(x)
        x, _ = self.RNN(x)
        # print(f'RNN后:{x.shape}')
        x = self.relu2(x)
        x = self.output_f(x)
        # print(f'Feature输出:{x.shape}')
        # x = self.relu3(x)
        # x = x.permute(0,2,1)
        # x = self.output_l(x)
        # x = x.permute(0,2,1)
        # print(f'len输出:{x.shape}')
        # return x.squeeze(-1)
        return x[:,-1,:].squeeze(-1)
    
class TN(nn.Module):
    def __init__(self,num_inputs) -> None:
        super(TN, self).__init__()
        self.TCN = TemporalConvNet(num_inputs, num_channels=[48], kernel_size=3, dropout=0.5)   # 48, 3, 0.5
        self.SA =  ExternalAttention(48, 24)                                                    # 48, 24
        self.relu1 = nn.ReLU(inplace=False)
        self.RNN = nn.LSTM(48, 24, 1)                                               # 48, 24, 1,0.3 LSTM
        self.relu2 = nn.ReLU(inplace=False)
        self.output_f = nn.Linear(24, 1)                                                        # 24, 1
        # self.relu3 = nn.ReLU(inplace=True)
        # self.output_l = nn.Linear(20, 1)
        # self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):     # 32, 20, 12  (Bath size, channel, length)
        # print(f'输入:{x.shape}')
        x = x.permute(0,2,1)  # 32, 12, 20
        # print(f'交换:{x.shape}')
        x = self.TCN(x)   
        # print(f'TCN后:{x.shape}')
        x = x.permute(0,2,1)   
        # print(f'交换:{x.shape}')
        x = self.SA(x)
        # print(f'SA后:{x.shape}')
        x = self.relu1(x)
        x, _ = self.RNN(x)
        # print(f'RNN后:{x.shape}')
        x = self.relu2(x)
        x = self.output_f(x)
        # print(f'Feature输出:{x.shape}')
        # x = self.relu3(x)
        # x = x.permute(0,2,1)
        # x = self.output_l(x)
        # x = x.permute(0,2,1)
        # print(f'len输出:{x.shape}')
        # return x.squeeze(-1)
        return x[:,-1,:].squeeze(-1)
    
class TP(nn.Module):
    def __init__(self,num_inputs) -> None:
        super(TP, self).__init__()
        self.TCN = TemporalConvNet(num_inputs, num_channels=[48], kernel_size=3, dropout=0.5)   # 48, 3, 0.5
        self.SA =  ExternalAttention(48, 24)                                                    # 48, 24
        self.relu1 = nn.ReLU(inplace=False)
        self.RNN = nn.LSTM(48, 24, 1)                                               # 48, 24, 1,0.3 LSTM
        self.relu2 = nn.ReLU(inplace=False)
        self.output_f = nn.Linear(24, 1)                                                        # 24, 1
        # self.relu3 = nn.ReLU(inplace=True)
        # self.output_l = nn.Linear(20, 1)
        # self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):     # 32, 20, 12  (Bath size, channel, length)
        # print(f'输入:{x.shape}')
        x = x.permute(0,2,1)  # 32, 12, 20
        # print(f'交换:{x.shape}')
        x = self.TCN(x)   
        # print(f'TCN后:{x.shape}')
        x = x.permute(0,2,1)   
        # print(f'交换:{x.shape}')
        x = self.SA(x)
        # print(f'SA后:{x.shape}')
        x = self.relu1(x)
        x, _ = self.RNN(x)
        # print(f'RNN后:{x.shape}')
        x = self.relu2(x)
        x = self.output_f(x)
        # print(f'Feature输出:{x.shape}')
        # x = self.relu3(x)
        # x = x.permute(0,2,1)
        # x = self.output_l(x)
        # x = x.permute(0,2,1)
        # print(f'len输出:{x.shape}')
        # return x.squeeze(-1)
        return x[:,-1,:].squeeze(-1)
    
class WT(nn.Module):
    def __init__(self,num_inputs) -> None:
        super(WT, self).__init__()
        self.TCN = TemporalConvNet(num_inputs, num_channels=[48], kernel_size=3, dropout=0.5)   # 48, 3, 0.5
        self.SA =  ExternalAttention(48, 24)                                                    # 48, 24
        self.relu1 = nn.ReLU(inplace=False)
        self.RNN = nn.LSTM(48, 24, 1)                                               # 48, 24, 1,0.3 LSTM
        self.relu2 = nn.ReLU(inplace=False)
        self.output_f = nn.Linear(24, 1)                                                        # 24, 1
        # self.relu3 = nn.ReLU(inplace=True)
        # self.output_l = nn.Linear(20, 1)
        # self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):     # 32, 20, 12  (Bath size, channel, length)
        # print(f'输入:{x.shape}')
        x = x.permute(0,2,1)  # 32, 12, 20
        # print(f'交换:{x.shape}')
        x = self.TCN(x)   
        # print(f'TCN后:{x.shape}')
        x = x.permute(0,2,1)   
        # print(f'交换:{x.shape}')
        x = self.SA(x)
        # print(f'SA后:{x.shape}')
        x = self.relu1(x)
        x, _ = self.RNN(x)
        # print(f'RNN后:{x.shape}')
        x = self.relu2(x)
        x = self.output_f(x)
        # print(f'Feature输出:{x.shape}')
        # x = self.relu3(x)
        # x = x.permute(0,2,1)
        # x = self.output_l(x)
        # x = x.permute(0,2,1)
        # print(f'len输出:{x.shape}')
        # return x.squeeze(-1)
        return x[:,-1,:].squeeze(-1)