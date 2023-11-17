import torch
import torch.nn as nn
import numpy as np
from models.LSTM.lstm import LSTMmodel
from models.CNN_LSTM.cnn_lstm import CNN_LSTM_Model
from models.RNN.rnn import RNNmodel
from models.CNN.cnn import CNNmodel
from models.GRU.gru import GRUmodel
from models.SEQ2SEQ.seq2seq import Encoder, Decoder, Seq2Seq
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
data_path = "./data/BRT/厦门北站_brtdata.npz"       # 数据集路径
data_name = "厦门北站"                              # 站点数据集名称
save_mod_dir = "./models/save"                      # 模型保存路径

'''load_data'''
def load_data(cut_point, data_path, input_length=7*24, output_length=1*24, days=61, hours=24):
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
    
    seq_days = np.arange(days)
    seq_hours = np.arange(hours)
    seq_day_hour = np.transpose(
        [np.repeat(seq_days, len(seq_hours)),
         np.tile(seq_hours, len(seq_days))]
    )   # Cartesian Product

    # 按照列方向拼接 --> [客流量, days, hours]
    data = np.concatenate((data, seq_day_hour), axis=1)

    train_data = data[:cut_point]
    test_data = data[cut_point:]

    # 分别进行min-max scaling
    train_data, min_train_data, max_train_data = min_max_normalise_numpy(train_data)
    test_data, min_test_data, max_test_data = min_max_normalise_numpy(test_data)

    '''
        input_data  模型输入 连续 input_length  // 24 = 7天的数据
        output_data 模型输出 未来 output_length // 24 = 1天的数据
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
    model = CNN_LSTM_Model(input_size=input_size, output_size=output_size).to(device)

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

def seq2seq(params):
    global lr, epochs, model
    '''超参数加载'''
    input_size = params["input_size"]               # 输入特征数
    hidden_size = params["hidden_size"]             # LSTM隐藏层神经元数
    num_layers = params["num_layers"]               # LSTM层数
    output_size = params["output_size"]             # 输出特征数
    lr = params["lr"]                               # 学习率
    epochs = params["epochs"]                       # 训练轮数
    encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    decoder = Decoder(output_size=output_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

def train(model_name, save_mod=False):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []      # 保存训练过程中loss数据
    # 开始计时
    start_time = time.time()  # 记录模型训练开始时间

    # train model
    for epoch in range(epochs):
        optimizer.zero_grad()   # 在每个epoch开始时清零梯度
        if(model_name=="SEQ2SEQ"):
            Y_hat = model(X,Y,1)
        else:
            Y_hat = model(X)
        # print('Y_hat.size: [{},{},{}]'.format(Y_hat.size(0), Y_hat.size(1), Y_hat.size(2)))
        # print('Y.size: [{},{},{}]'.format(Y.size(0), Y.size(1), Y.size(2)))
        loss = loss_function(Y_hat, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=4, norm_type=2)
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], MSE-Loss: {loss.item()}')
        loss_list.append(loss.item())
    end_time = time.time()  # 记录模型训练结束时间
    total_time = end_time - start_time  # 计算总耗时
    print(f"本次模型训练总耗时: {total_time} 秒，超越了全国99.2%的单片机，太棒啦！")
    if save_mod == True:
        # save model -> ./**/model_name_eopch{n}.pth
        torch.save(model.state_dict(), '{}/{}_epoch{}.pth'.format(save_mod_dir, model_name, epochs))
        print("Trained Model have saved in:", '{}/{}_epoch{}.pth'.format(save_mod_dir, model_name, epochs))

    # save training loss graph
    epoch_index = list(range(1,epochs+1))
    plt.figure(figsize=(10, 6))
    sns.set(style='darkgrid')
    plt.plot(epoch_index, loss_list, marker='.', label='MSE Loss', color='red')
    plt.xlabel('Epoch')  # x 轴标签
    plt.ylabel('MSE Loss')  # y 轴标签
    plt.title(f'{data_name}车站,{model_name}模型,训练过程MSE Loss曲线图',fontproperties='SimHei', fontsize=20)
    plt.legend()
    plt.grid(True)  # 添加网格背景
    plt.savefig('./img/{}_{}_training_MSEloss_epoch{}.png'.format(data_name, model_name, epochs))
    plt.close()

def predict(model_name):
    # predict
    with torch.no_grad():
        if(model_name=="SEQ2SEQ"):
            # 关闭教师强制策略
            testY_hat = model(testX, testY, teacher_forcing_ratio=0)
        else:
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
    # predict = inverse_min_max_normalise_numpy(predict, min_vals=min_test_data[:,0], max_vals=max_test_data[:,0])
    # real = inverse_min_max_normalise_numpy(real, min_vals=min_test_data[:,0], max_vals=max_test_data[:,0])
    predict = inverse_min_max_normalise_tensor(testY_hat, min_vals=min_test_data, max_vals=max_test_data)
    real = inverse_min_max_normalise_tensor(testY, min_vals=min_test_data, max_vals=max_test_data)
    # draw
    predict = predict[0, :, 0].cpu().data.numpy()
    real = real[0, :, 0].cpu().data.numpy()
    # 生成时间序列，时间间隔为1小时，共24小时
    time_series = list(range(24))
    # 定义时间标签
    time_labels = [f"{i}:00" for i in range(24)]
    font = FontProperties(family='SimHei', size=20)
    plt.figure(figsize=(12, 6))
    sns.set(style='darkgrid')
    plt.ylabel('客流量', fontproperties=font)
    plt.plot(time_series, predict, marker='o', color='red', label='预测值')
    plt.plot(time_series, real, marker='o', color='blue', label='真实值')
    plt.xlabel('时间', fontproperties=font)
    plt.ylabel('客流量', fontproperties=font)
    plt.title(f'{data_name}车站2023年5月31日客流量,{model_name}模型,预测值效果对比图(epoch={epochs})', fontproperties=font)
    plt.legend(prop=font)
    plt.xticks(time_series, time_labels)  # 设置时间标签
    plt.savefig('./img/{}_{}_prediction_epoch{}.png'.format(data_name, model_name, epochs))
    plt.show()

def main(model_name, save_mod):
    '''加载数据集'''
    global X, Y, testX, testY
    X, Y, testX, testY = load_data(cut_point=1272, data_path=data_path)

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
    elif model_name == "SEQ2SEQ":
        seq2seq(params)

    train(model_name=model_name, save_mod=save_mod)
    predict(model_name=model_name)

if __name__ == '__main__':
    '''CHOOSE DATAPATH'''
    # (1464, 45, 1) ——> 45个BRT站点连续61days共计1464个小时的客流量数据
    station_list = ['第一码头', '开禾路口', '思北', '斗西路',
                    '二市', '文灶', '金榜公园', '火车站',
                    '莲坂', '龙山桥', '卧龙晓城', '东芳山庄',
                    '洪文', '前埔枢纽站', '蔡塘', '金山', '市政务服务中心',
                    '双十中学', '县后', '高崎机场', 'T4候机楼', '嘉庚体育馆',
                    '诚毅学院', '华侨大学', '大学城', '产业研究院', '中科院',
                    '东宅', '田厝', '厦门北站', '凤林', '东安', '后田', '东亭',
                    '美峰', '蔡店', '潘涂', '滨海新城西柯枢纽', '官浔', '轻工食品园',
                    '四口圳', '工业集中区', '第三医院', '城南', '同安枢纽']
    # 创建一个字典，将数字与车站名称进行映射
    station_dict = {i: station_list[i-1] for i in range(1, len(station_list)+1)}
    # 创建 DataFrame
    df = pd.DataFrame(list(station_dict.items()), columns=['编号', '车站名称'])
    # 设置显示宽度以保持对齐
    pd.set_option('display.max_colwidth', 28)
    print(df.to_string(index=False))
    user_input = input("请输入你要训练的车站数据集编号（1~45，否则随机选择）：")
    if int(user_input) in station_dict:
        data_name = station_dict[int(user_input)]
    else:
        # 随机选择一个站点(1-45之间的一个站点)
        random_station = np.random.randint(1, 46)
        data_name = station_list[random_station]
    data_path = f"./data/BRT/{data_name}_brtdata.npz"
    print("数据加载中, 请稍后……", data_path)

    '''INPUT YOUR MODEL NAME'''
    name_list = ["CNN", "RNN", "LSTM", "CNN_LSTM", "GRU", "Attention_LSTM", "SEQ2SEQ"]
    model_name = input("请输入要使用的模型【1: CNN   2: RNN   3: LSTM   4: CNN_LSTM   5: GRU   6: Attention_LSTM   7: SEQ2SEQ】\n")
    if model_name.isnumeric() and int(model_name) <= len(name_list):
        model_name = name_list[int(model_name) - 1]
    '''SAVE MODE'''
    save_mod = input("是否要保存训练后的模型？（输入 '1' 保存，否则不保存）\n")
    if int(save_mod) == 1:
        save_mod = True
    else:
        save_mod = False
    '''main()'''
    main(model_name, save_mod)