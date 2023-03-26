import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F  # 激励函数都在这
import torch.utils.data as Data  #
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 获取训练集-------------------------------------------------------------------
def get_train_data(train_data, time_step=3):
    train_x, train_y = [], []  # 训练集
    for i in range(len(train_data) - time_step):
        x = train_data[i:i + time_step]
        y = train_data[i + time_step]
        train_x.append(x)
        train_y.append(y)
    return np.asarray(train_x), np.asarray(train_y), train_data

def average(loss):
    loss_a = []
    av_len = 40
    for index in range(len(loss)-av_len):
        loss_a.append(np.mean(loss[index:index+av_len]))
    return loss_a

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(in_ch, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        h = self.group_norm(x)
        # h = x
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        # print((x+h).shape)
        # assert 1==2
        return x + h
    
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)
# (h_in - 1)/2  + 1
    def forward(self, x):
        x = self.c1(x) + self.c2(x)
        return x

class ATTENTION(nn.Module):
    def __init__(self):
        super(ATTENTION, self).__init__()
        self.linear1 = nn.Linear(1, 64)
        self.ch = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.attn1 = AttnBlock(64)
        self.ch2 = nn.Conv2d(64, 8, 3, stride=1, padding=1)
        self.linear2 = nn.Linear(8*3*64, 4*3*64)
        self.linear3 = nn.Linear(4*3*64, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self, x):
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        x = self.linear1(x)
        x = self.ch(x.unsqueeze(1))
        x = self.attn1(x)
        x = self.ch2(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x).squeeze()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(3, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self, x):
        if x.shape[-1] == 1:
            x = x.squeeze()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x).squeeze()

class TransformerRegression(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # 定义Transformer的编码器
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 定义Transformer的解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 定义全连接层
        self.fc_in = nn.Linear(input_dim, d_model)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (seq_len, batch_size)
        src = self.fc_in(x.view(x.shape[0], x.shape[1], 1).permute(1, 0, 2))  # (batch_size, seq_len, input_dim)

        # 编码
        encoded = self.transformer_encoder(src)

        # 解码
        decoded = self.transformer_decoder(src, encoded)

        # 全连接层映射
        out = self.fc(decoded[-1])  # 取解码结果的最后一个时间步作为输出

        return out.squeeze()


BATCH_SIZE = 32
class MLE(object):
    def __init__(self, x_t, device, actor_lr, model_type='Attention'):
        self.device = device
        self.x_t0 = x_t    
        self.model_type = model_type 
        if model_type == 'MLP':
            self.model = MLP().to(self.device)
        elif model_type == 'Attention':
            self.model = ATTENTION().to(self.device)
        elif model_type == 'Trans':
            self.model = TransformerRegression(input_dim=1, output_dim=1).to(self.device)
        else:
            self.model = torch.nn.LSTM(input_size=1, hidden_size=1, num_layers=3, batch_first=True).to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=actor_lr)
        
    def train(self, loader, EPOCH=100):
        loss_dict = []
        for train_index in range(EPOCH):
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if self.model_type == 'LSTM':
                    prediction, _ = self.model(batch_x)
                    loss_mse = self.loss_func(prediction[:, -1, :], batch_y) / BATCH_SIZE 
                else:
                    prediction = self.model(batch_x)
                    loss_mse = self.loss_func(prediction, batch_y) / BATCH_SIZE  # 计算两者的误差
                self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
                loss_mse.backward()  # 误差反向传播, 计算参数更新值
                self.optimizer.step()
            print(f"epoch: {train_index}, loss: {loss_mse}")
            loss_dict.append(loss_mse.detach().cpu().item())
        return loss_dict
    
    def predict(self, data_):
        if self.model_type == 'LSTM':
            a_t, _ = self.model(data_.to(self.device))
            a_t = a_t[:, -1, :]
        else:
            a_t = self.model(data_.to(self.device))
        return a_t.detach().cpu().numpy()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
print(torch.cuda.is_available())
print(torch.__version__)
torch.manual_seed(0)
data_dict = pd.read_excel('./数据（公开）.xlsx', usecols=[1], header=None)
data = np.array(data_dict)[:,0]
data_mean, data_std = data.mean(), data.std()
data = (data - data_mean) / data_std
data_s, data_sn, data = get_train_data(data)
data_s = data_s[:, :, np.newaxis] 
data_s = torch.from_numpy(data_s).float()
data_sn = torch.from_numpy(data_sn).float()
# 将训练集先转换成torch能识别的Dataset
torch_dataset = Data.TensorDataset(data_s[:], data_sn[:])
# 把Dataset放入Dataloader
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
model_type = ['MLP', 'Attention', 'LSTM', 'Trans']
model_num = 0
mle = MLE(data, device, 1e-4, model_type[model_num])
loss_dict = mle.train(loader=loader, EPOCH=1000)
torch.save(mle, f"./model/{model_type[model_num]}.file")
mle = torch.load(f"./model/{model_type[model_num]}.file")
print("model loaded!")
pre = mle.predict(data_s[:])
pre = pre*data_std + data_mean
data = data*data_std + data_mean
np.savez(f"./results/{model_type[model_num]}.npz", pre=pre, loss_dict=loss_dict, 
         pre_last=pre[-1], data_last=data[-1])
# 绘制训练损失函数图像：
print(f"2022-12-27 truth: {data[-1]}, prediction: {pre[-1]}")

# 绘制预测图像：
plt.figure()
plt.plot(np.array(pre), label='prediction', color='b')
# plt.plot(np.linspace(303, data.shape[0], 71), np.array(pre), label='prediction', color='b')
plt.plot(np.array(data), label='label', color='r')
plt.legend()
plt.xlabel("day")
plt.ylabel("value")
plt.title("Prediction Figure")
plt.savefig(f"./figure/{model_type[model_num]}_Pre.png")
plt.show()
# 绘制训练损失函数图像：
plt.figure()
plt.plot(np.array(loss_dict), label='Loss', color='b')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale('log')
plt.title("Loss Figure")
plt.savefig(f"./figure/{model_type[model_num]}_Loss.png")
plt.show()

# 绘制所有损失函数图像
loss1 = np.load(f"./results/{model_type[0]}.npz")['loss_dict']
loss2 = np.load(f"./results/{model_type[1]}.npz")['loss_dict']
loss3 = np.load(f"./results/{model_type[2]}.npz")['loss_dict']
loss4 = np.load(f"./results/{model_type[3]}.npz")['loss_dict']
plt.figure()
plt.plot(average(loss1), label=f'Loss_{model_type[0]}', color='b')
plt.plot(average(loss2), label=f'Loss_{model_type[1]}', color='r')
plt.plot(average(loss3), label=f'Loss_{model_type[2]}', color='g')
plt.plot(average(loss4), label=f'Loss_{model_type[3]}')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale('log')
plt.title("Loss Figure")
plt.savefig(f"./figure/Loss_all.png")
plt.show()
