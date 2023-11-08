import torch
import torch.nn as nn
import numpy as np
from models.LSTM.lstm import LSTMmodel
from models.CNN_LSTM.cnn_lstm import CNN_LSTM_Model
from models.RNN.rnn import RNNmodel
from models.CNN.cnn import CNNmodel
from models.GRU.gru import GRUmodel
from models.AttentionLSTM.attention_lstm import AttentionLSTMmodel
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import re
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = None
Y = None
testX = None
testY = None
lr = 0.0001
epochs = 100
model = None
data_path = "./data/PEMS03/new_pems03.npz"      # 数据集路径
# 5760 = 5min * 12 * 24 * 20 = 20天
save_mod_dir = "./models/save"                      # 模型保存路径

'''load_data'''
def load_data(cut_point, data_path, input_length=12*24*7, output_length=12*24):
    '''
        INPUT:
                cut_point, 训练和测试数据集划分点
                days, 客流数据的总天数
                hours, 每天小时数
                input_length, 模型输入的过去7天的数据作为一个序列
                output_length, 预测未来1天的数据
        OUTPUT:
                train_data
                test_data
    '''
    data = np.load(data_path)['data']
    data = data.astype(np.float32)
    data = data[:, np.newaxis]

    train_data = data[:cut_point]
    test_data = data[cut_point:]

    # 分别进行min-max scaling
    train_data, min_train_data, max_train_data = min_max_normalise_numpy(train_data)
    test_data, min_test_data, max_test_data = min_max_normalise_numpy(test_data)

    '''
        input_data 模型输入 连续 input_length//7 天的数据
        output_data 模型输出 未来 output_length//7 天的数据
    '''
    input_data = []
    output_data = []

    # 滑动窗口法
    for i in range(len(train_data) - input_length - output_length + 1):
        input_seq = train_data[i : i + input_length]
        output_seq = train_data[i + input_length : i + input_length + output_length, 0]

        input_data.append(input_seq)
        output_data.append(output_seq)

    # 转为torch.tensor
    X = torch.tensor(input_data, dtype=torch.float32, device=device)
    Y = torch.tensor(output_data, dtype=torch.float32, device=device)
    Y = Y.unsqueeze(-1)     # 在最后一个维度（维度索引为1）上增加一个维度
    # X = torch.tensor([item.cpu().detach().numpy() for item in input_data], dtype=torch.float32, device=device)
    # Y = torch.tensor([item.cpu().detach().numpy() for item in output_data], dtype=torch.float32, device=device)

    test_inseq = []
    test_outseq = []
    # 滑动窗口法
    for i in range(len(test_data) - input_length - output_length + 1):
        input_seq = test_data[i : i + input_length]
        output_seq = test_data[i + input_length : i + input_length + output_length, 0]

        test_inseq.append(input_seq)
        test_outseq.append(output_seq)

    # 转为torch.tensor
    testX = torch.tensor(test_inseq, dtype=torch.float32, device=device)
    testY = torch.tensor(test_outseq, dtype=torch.float32, device=device)
    testY = testY.unsqueeze(-1)     # 在最后一个维度（维度索引为1）上增加一个维度
    # testX = torch.tensor([item.cpu().detach().numpy() for item in test_inseq], dtype=torch.float32, device=device)
    # testY = torch.tensor([item.cpu().detach().numpy() for item in test_outseq], dtype=torch.float32, device=device)

    # 输出数据形状
    print("traindata - Input shape:", X.shape)
    print("traindata - Output shape:", Y.shape)
    print("testdata - Input shape:", testX.shape)
    print("testdata - Output shape:", testY.shape)
    return X, Y, testX, testY

'''min-max Scaling'''
def min_max_normalise_numpy(x):
    # shape: [sequence_length, features]
    min_vals = np.min(x, axis=0)
    max_vals = np.max(x, axis=0)
    # [features] -> shape: [1, features]
    min_vals = np.expand_dims(min_vals, axis=0)
    max_vals = np.expand_dims(max_vals, axis=0)
    # 归一化 -> [-1, 1]
    normalized_data = 2 * (x - min_vals) / (max_vals - min_vals) - 1
    return normalized_data, min_vals, max_vals

def min_max_normalise_tensor(x):
    # shape: [samples, sequence_length, features]
    min_vals = x.min(dim=1).values.unsqueeze(1)
    max_vals = x.max(dim=1).values.unsqueeze(1)
    # data ->[-1, 1]
    normalise_x = 2 * (x - min_vals) / (max_vals - min_vals) - 1
    return normalise_x, min_vals, max_vals

def cnn(params):
    global lr, epochs, model
    '''超参数加载'''
    input_size = params["input_size"]           # 输入特征数
    hidden_size = params["hidden_size"]         # CNN output_channels
    output_size = params["output_size"]         # 输出特征数
    lr = params["lr"]                           # 学习率
    epochs = params["epochs"]                   # 训练轮数
    model = CNNmodel(input_size, hidden_size, output_size).to(device)

def rnn(params):
    global lr, epochs, model
    '''超参数加载'''
    input_size = params["input_size"]           # 输入特征数
    hidden_size = params["hidden_size"]         # RNN隐藏层神经元数
    num_layers = params["num_layers"]           # RNN层数
    output_size = params["output_size"]         # 输出特征数
    bidirectional = params["bidirectional"].lower() == "true"   # 是否双向
    lr = params["lr"]                           # 学习率
    epochs = params["epochs"]                   # 训练轮数
    model = RNNmodel(input_size, hidden_size, num_layers, output_size, bidirectional=bidirectional).to(device)

def lstm(params):
    global lr, epochs, model
    '''超参数加载'''
    input_size = params["input_size"]               # 输入特征数
    hidden_size = params["hidden_size"]             # LSTM隐藏层神经元数
    num_layers = params["num_layers"]               # LSTM层数
    bidirectional = params["bidirectional"].lower() == "true"   # 是否双向
    output_size = params["output_size"]             # 输出特征数
    lr = params["lr"]                               # 学习率
    epochs = params["epochs"]                       # 训练轮数

    model = LSTMmodel(input_size, hidden_size, num_layers, output_size, bidirectional=bidirectional).to(device)

def cnn_lstm(params):
    global lr, epochs, model
    '''超参数加载'''
    input_size = params["input_size"]           # 输入特征数
    output_size = params["output_size"]         # 输出特征数
    lr = params["lr"]                           # 学习率
    epochs = params["epochs"]                   # 训练轮数
    model = CNN_LSTM_Model(input_size=1, output_size=output_size, output_length=24*12).to(device)

def gru(params):
    global lr, epochs, model
    '''超参数加载'''
    input_size = params["input_size"]               # 输入特征数
    hidden_size = params["hidden_size"]             # LSTM隐藏层神经元数
    num_layers = params["num_layers"]               # LSTM层数
    bidirectional = params["bidirectional"].lower() == "true"   # 是否双向
    output_size = params["output_size"]             # 输出特征数
    lr = params["lr"]                               # 学习率
    epochs = params["epochs"]                       # 训练轮数
    model = GRUmodel(input_size, hidden_size, num_layers, output_size, bidirectional=bidirectional).to(device)

def attention_lstm(params):
    global lr, epochs, model
    '''超参数加载'''
    input_size = params["input_size"]               # 输入特征数
    hidden_size = params["hidden_size"]             # LSTM隐藏层神经元数
    num_layers = params["num_layers"]               # LSTM层数
    bidirectional = params["bidirectional"].lower() == "true"   # 是否双向
    output_size = params["output_size"]             # 输出特征数
    lr = params["lr"]                               # 学习率
    epochs = params["epochs"]                       # 训练轮数
    model = AttentionLSTMmodel(input_size, hidden_size, num_layers, output_size, bidirectional=bidirectional).to(device)

def train(model_name, save_mod=False):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_list = []      # 保存训练过程中loss数据

    # train model
    for epoch in range(epochs):
        Y_hat = model(X)
        loss = loss_function(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
        loss_list.append(loss.item())

    if save_mod == True:
        
        # save model -> ./**/model_name_eopch{n}.pth
        torch.save(model.state_dict(), '{}/{}_epoch{}.pth'.format(save_mod_dir, model_name, epochs))
        print("Trained Model have saved in:", '{}/{}_epoch{}.pth'.format(save_mod_dir, model_name, epochs))

    # save training loss graph
    epoch_index = list(range(1,epochs+1))
    plt.plot(epoch_index, loss_list)
    plt.xlabel('Epoch')  # x 轴标签
    plt.ylabel('MSE Loss')  # y 轴标签
    plt.savefig('./img/{}_training_MSEloss_epoch{}.png'.format(model_name, epochs))
    plt.close()

def continue_train(model_name, save_mod=True):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    los_list = []      # 保存训练过程中loss数据

    # 设置之前保存模型的路径
    saved_model_path = './models/save/CNN_LSTM_epoch50.pth'
    # 从文件名中提取数字作为新的 epoch 起点
    epoch_number = re.findall(r'\d+', saved_model_path)
    if epoch_number:
        new_start_epoch = int(epoch_number[0]) + 1  # 获取数字并加1作为新的起点
        print("New epoch starting point:", new_start_epoch)
    else:
        print("Epoch number not found in the file name.")

    # 加载模型权重
    model.load_state_dict(torch.load(saved_model_path))

    # train model
    epochs = new_start_epoch + epochs
    for epoch in range(new_start_epoch, epochs):
        Y_hat = model(X)
        loss = loss_function(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
        loss_list.append(loss.item())

    if save_mod == True:
        # save model -> ./**/model_name_eopch{n}.pth
        torch.save(model.state_dict(), '{}/{}_epoch{}.pth'.format(save_mod_dir, model_name, epochs))
        print("Trained Model have saved in:", '{}/{}_epoch{}.pth'.format(save_mod_dir, model_name, epochs))

    # save training loss graph
    epoch_index = list(range(new_start_epoch, epochs))
    plt.plot(epoch_index, loss_list)
    plt.xlabel('Epoch')  # x 轴标签
    plt.ylabel('MSE Loss')  # y 轴标签
    plt.savefig('./img/{}_training_MSEloss_epoch{}.png'.format(model_name, epochs))
    plt.close()

def predict(model_name):
    # predict
    with torch.no_grad():
        testY_hat = model(testX)
    # cal_loss
    MAEloss = nn.L1Loss()
    MSEloss = nn.MSELoss()
    
    # MAE:平均绝对误差
    mae_val = MAEloss(testY_hat, testY)
    # MSE:均方误差
    mse_val = MSEloss(testY_hat, testY)
    # RMSE:均方根误差
    rmse_val = torch.sqrt(mse_val)

    losses = [
        {'Loss Type': 'MAE Loss', 'Loss Value': mae_val.cpu().item()},
        {'Loss Type': 'MSE Loss', 'Loss Value': mse_val.cpu().item()},
        {'Loss Type': 'RMSE Loss', 'Loss Value': rmse_val.cpu().item()}
    ]
    # losses_df = pd.DataFrame(list(losses.items()), columns=['Loss Type', 'Loss Value'])
    losses_df = pd.DataFrame(losses)
    print(losses_df)

    # draw
    predict = testY_hat[0, :, 0].cpu().data.numpy()
    real = testY[0, :, 0].cpu().data.numpy()
    plt.plot(predict, 'r', label='predict')
    plt.plot(real, 'b', label='real')
    plt.legend(loc='best')
    plt.savefig('./img/{}_prediction_epoch{}.png'.format(model_name, epochs))
    plt.pause(4)

def main(model_name):
    '''加载数据集'''
    global X, Y, testX, testY
    X, Y, testX, testY = load_data(cut_point=3168, data_path=data_path)

    '''读取超参数配置文件'''
    params = None
    with open(f"./config/{model_name.lower()}_params.json", 'r', encoding='utf-8') as file:
        params = json.load(file)
    
    if model_name == "RNN":
        rnn(params)
    elif model_name == "LSTM":
        lstm(params)
    elif model_name == "CNN_LSTM":
        cnn_lstm(params)
    elif model_name == "GRU":
        gru(params)
    elif model_name == "CNN":
        cnn(params)
    elif model_name == "Attention_LSTM":
        attention_lstm(params)


    # train(model_name=model_name, save_mod=True)
    continue_train(model_name=model_name)
    predict(model_name=model_name)

if __name__ == '__main__':
    '''INPUT YOUR MODEL NAME'''
    name_list = ["CNN", "RNN", "LSTM", "CNN_LSTM", "GRU", "Attention_LSTM"]
    model_name = input("请输入模型名字【1: CNN   2: RNN   3: LSTM   4: CNN_LSTM   5: GRU   6: Attention_LSTM】\n")
    if model_name.isnumeric() and int(model_name) <= len(name_list):
        model_name = name_list[int(model_name) - 1]

    '''main()'''
    main(model_name)