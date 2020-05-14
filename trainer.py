import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable
from models.darnn import AttnEncoder, AttnDecoder
from data.dataset import MyDataSet
from data.dataset_op import op
import config

# STEP.1:导入原始数据集
demo=MyDataSet('C:/Users/dell/Desktop/train2(3).csv')

# STEP.2:链接数据操作类
demo_op = op(demo.df)
print(demo_op.test())

# STEP.3:划分time_step:10个点到1个点的映射，也就是sequence序列的长度为10
# data[0] is
# data[1] is
data = demo_op.divide_into_XY()
print(np.array(data[0]).shape[1])

# STEP.4:数据x,y的归一化
x = demo_op.preprocessor(input_data = np.array(data[0]))
print(x)
y = demo_op.preprocessor(input_data = np.reshape(data[1], (len(data[1]), 1)))
print(y)

# STEP.5:划分训练集和测试集
train_size = int(len(y) * 0.8)
test_size = int(len(y)-train_size)
train_X = x[:train_size]
train_Y = y[:train_size]
test_X = x[train_size:]
test_Y = y[train_size:]

# 6: load net
encoder = AttnEncoder(input_size=demo_op.get_num_features(), hidden_size=config.ENCODER_HIDDEN_SIZE, time_step=config.time_step)
decoder = AttnDecoder(code_hidden_size=config.ENCODER_HIDDEN_SIZE, hidden_size=config.DECODER_HIDDEN_SIZE, time_step=config.time_step)
encoder_optim = torch.optim.Adam(encoder.parameters(), config.lr)
decoder_optim = torch.optim.Adam(decoder.parameters(), config.lr)
loss_func = nn.MSELoss()

# 7: start train
def to_variable(x):
    return Variable(torch.from_numpy(x).float())

print('start')
for epoch in range(config.num_epochs):
    i = 0
    loss_sum = 0
    while(i < train_size):
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        batch_end = i + config.batch_size
        if(batch_end >= train_size):
            batch_end = train_size
        var_x = to_variable(train_X[i:batch_end])
        var_y = to_variable(train_Y[i:batch_end])
        if var_x.dim() == 2:
            var_x = var_x.unsqueeze(2)
        code = encoder(var_x)
        y_res = decoder(code, var_x)
        loss = loss_func(y_res, var_y)
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()
        loss_sum += loss.item()
        i = batch_end
    print('epoch [%d] finished, the average loss is %f' % (epoch, loss_sum))
    if(epoch + 1)%(config.interval) == 0 or epoch + 1 == config.num_epochs:
        torch.save(encoder.state_dict(), 'models/encoder'+str(epoch+1)+'-norm'+'.model')
        torch.save(decoder.state_dict(), 'models/decoder' + str(epoch + 1) + '-norm' + '.model')

# 8: test
    def predict(x):
        y_pred = np.zeros(x.shape[0])
        i = 0
        while (i < x.shape[0]):
            batch_end = i + config.batch_size
            if batch_end > x.shape[0] :
                batch_end = x.shape[0]
            var_x_input = to_variable(x[i:batch_end])
            if var_x_input.dim() == 2:
                var_x_input = var_x_input.unsqueeze(2)
            code = encoder(var_x_input)
            y_res = decoder(code, var_x_input)
            for j in range(i, batch_end):
                y_pred[j] = y_res[j - i, -1]
            i = batch_end
        return y_pred

    y_pred_train = predict(train_X)
    y_pred_test = predict(test_X)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(range(2000,train_size),train_Y[2000:], label='train truth', color='black')
    plt.plot(range(train_size, train_size + test_size),test_Y, label='ground truth', color='black')
    plt.plot(range(2000, train_size), y_pred_train[2000:], label='predicted train', color='red')
    plt.plot(range(train_size, train_size + test_size), y_pred_test, label='predicted test',
             color='blue')
    plt.savefig('results/res-' + str(config.num_epochs) + '-' + str(config.batch_size) + '.png')




