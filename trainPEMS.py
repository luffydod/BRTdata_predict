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
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os
import json
import time
import pandas as pd
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = None
Y = None
testX = None
testY = None
lr = 0.0001
epochs = 100
model = None
min_test_data = None
max_test_data = None                                # 测试集的最大最小值

data_path = "./data/PEMS03/new_pems03_num88.npz"      # 数据集路径
# 5760 = 5min * 12 * 24 * 20 = 20天
save_mod_dir = "./models/save"                      # 模型保存路径

'''load_data'''
def load_data(cut_point, data_path, input_length=12*24*7, output_length=12*24):
    global min_test_data, max_test_data
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
    print("数据集处理完毕：")
    print("data - 原数据集 shape:", data.shape)
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
    # 归一化 -> [0, 1]
    # normalized_data = (x - min_vals) / (max_vals - min_vals)
    return normalized_data, min_vals, max_vals

def inverse_min_max_normalise_numpy(normalised_x, min_vals, max_vals):
    x = (normalised_x + 1) / 2 * (max_vals - min_vals) + min_vals
    return x

def min_max_normalise_tensor(x):
    # shape: [samples, sequence_length, features]
    min_vals = x.min(dim=1).values.unsqueeze(1)
    max_vals = x.max(dim=1).values.unsqueeze(1)
    # data ->[-1, 1]
    normalise_x = 2 * (x - min_vals) / (max_vals - min_vals) - 1
    return normalise_x, min_vals, max_vals

def inverse_min_max_normalise_tensor(x, min_vals, max_vals):
    min_vals = torch.tensor(min_vals).to(device)
    max_vals = torch.tensor(max_vals).to(device)
    # shape: [1, features] -> [1, 1, features]
    min_vals = min_vals.unsqueeze(0)
    max_vals = max_vals.repeat(x.shape[0], 1, 1)
    # [1, 1, features] -> [samples, 1, features]
    min_vals = min_vals.repeat(x.shape[0], 1, 1)
    max_vals = max_vals.repeat(x.shape[0], 1, 1)

    x = (x + 1) / 2 * (max_vals - min_vals) + min_vals
    return x

# MAPE: 平均绝对百分比误差
def MAPELoss(y_hat, y):
    x = torch.tensor(0.0001, dtype=torch.float32).to(device)
    y_new = torch.where(y==0, x, y)    # 防止分母为0
    abs_error = torch.abs((y - y_hat) / y_new)
    mape = 100. * torch.mean(abs_error)
    return mape

def cnn(params):
    global lr, epochs, model
    '''超参数加载'''
    input_size = params["input_size"]           # 输入特征数
    hidden_size = params["hidden_size"]         # CNN output_channels
    output_size = params["output_size"]         # 输出特征数
    lr = params["lr"]                           # 学习率
    epochs = params["epochs"]                   # 训练轮数
    model = CNNmodel(input_size, hidden_size, output_size, output_length=24*12).to(device)

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
    model = RNNmodel(input_size, hidden_size, num_layers, output_size, output_length=24*12, bidirectional=bidirectional).to(device)

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

    model = LSTMmodel(input_size, hidden_size, num_layers, output_size, output_length=24*12, bidirectional=bidirectional).to(device)

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
    model = GRUmodel(input_size, hidden_size, num_layers, output_size, output_length=24*12, bidirectional=bidirectional).to(device)

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
    model = AttentionLSTMmodel(input_size, hidden_size, num_layers, output_size, output_length=24*12, bidirectional=bidirectional).to(device)

def train(model_name, save_mod=False):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_list = []      # 保存训练过程中loss数据

    # 开始计时
    start_time = time.time()  # 记录模型训练开始时间
    # train model
    for epoch in range(epochs):
        Y_hat = model(X)
        loss = loss_function(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], MSE-Loss: {loss.item()}')
        loss_list.append(loss.item())
    end_time = time.time()  # 记录模型训练结束时间
    total_time = end_time - start_time  # 计算总耗时
    print(f"本次模型训练总耗时: {total_time} 秒，超越了全国99.2%的单片机，太棒啦！")
    if save_mod == True:
        # save model -> ./**/model_name_eopch{n}.pth
        torch.save(model.state_dict(), '{}/PEMS_{}_epoch{}.pth'.format(save_mod_dir, model_name, epochs))
        print("Trained Model have saved in:", '{}/{}_epoch{}.pth'.format(save_mod_dir, model_name, epochs))

    # save training loss graph
    epoch_index = list(range(1,epochs+1))
    plt.figure(figsize=(10, 6))
    sns.set(style='darkgrid')
    plt.plot(epoch_index, loss_list, marker='.', label='MSE Loss', color='red')
    plt.xlabel('Epoch')  # x 轴标签
    plt.ylabel('MSE Loss')  # y 轴标签
    plt.title(f'PEMS03数据集,{model_name}模型,训练过程MSE Loss曲线图',fontproperties='SimHei', fontsize=20)
    plt.legend()
    plt.grid(True)  # 添加网格背景
    plt.savefig('./img/PEMS03_{}_training_MSEloss_epoch{}.png'.format(model_name, epochs))
    plt.close()

def predict(model_name):
    # predict
    with torch.no_grad():
        testY_hat = model(testX)
    predict = testY_hat
    real = testY
    # cal_loss
    MAEloss = nn.L1Loss()
    MSEloss = nn.MSELoss()
    
    # MAE: 平均绝对误差
    mae_val = MAEloss(predict, real)
    # MAPE: 平均绝对百分比误差
    mape_val = MAPELoss(predict, real)
    # MSE:均方误差
    mse_val = MSEloss(predict, real)
    # RMSE:均方根误差
    rmse_val = torch.sqrt(mse_val)

    losses = [
        {'Loss Type': 'MAE Loss', 'Loss Value': mae_val.cpu().item()},
        {'Loss Type': 'MAPE Loss', 'Loss Value': mape_val.cpu().item()},
        {'Loss Type': 'RMSE Loss', 'Loss Value': rmse_val.cpu().item()}
    ]
    losses_df = pd.DataFrame(losses)
    print(losses_df)

    # predict = inverse_min_max_normalise_tensor(predict, min_vals=min_test_data, max_vals=max_test_data)
    # real = inverse_min_max_normalise_tensor(real, min_vals=min_test_data, max_vals=max_test_data)

    # draw
    predict_np = predict[10, :, 0].cpu().data.numpy()
    real_np = real[10, :, 0].cpu().data.numpy()

    predict_np = inverse_min_max_normalise_numpy(predict_np, min_vals=min_test_data[:,0], max_vals=max_test_data[:,0])
    real_np = inverse_min_max_normalise_numpy(real_np, min_vals=min_test_data[:,0], max_vals=max_test_data[:,0])

    font = FontProperties(family='SimHei', size=20)
    plt.figure(figsize=(12, 6))
    sns.set(style='darkgrid')
    plt.plot(predict_np, color='red', label='预测值')
    plt.plot(real_np, color='blue', label='真实值')
    plt.xlabel('时间', fontproperties=font)
    plt.ylabel('车流量', fontproperties=font)
    plt.title(f'PEMS03车流量数据集,{model_name}模型,预测值效果对比图(epoch={epochs})', fontproperties=font)
    plt.legend(prop=font)
    plt.savefig('./img/PEMS03_{}_prediction_epoch{}.png'.format(model_name, epochs))
    plt.show()

def main(model_name, save_mod):
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


    train(model_name=model_name, save_mod=save_mod)
    predict(model_name=model_name)

def continue_main(model_name):
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

    # 设置之前保存模型的路径
    saved_model_path = './models/save/PEMS_CNN_LSTM_epoch70.pth'
    # 加载模型权重
    model.load_state_dict(torch.load(saved_model_path))
    predict(model_name=model_name)

if __name__ == '__main__':
    '''INPUT YOUR MODEL NAME'''
    name_list = ["CNN", "RNN", "LSTM", "CNN_LSTM", "GRU", "Attention_LSTM"]
    model_name = input("请输入要使用的模型【1: CNN   2: RNN   3: LSTM   4: CNN_LSTM   5: GRU   6: Attention_LSTM】\n")
    if model_name.isnumeric() and int(model_name) <= len(name_list):
        model_name = name_list[int(model_name) - 1]
    '''SAVE MODE'''
    save_mod = input("是否要保存训练后的模型？（输入 '1' 保存，否则不保存）\n")
    if int(save_mod) == 1:
        save_mod = True
    else:
        save_mod = False
    
    # continue_main(model_name=model_name)
    '''main()'''
    main(model_name, save_mod)