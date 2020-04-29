from data.dataset_op import op
from data.dataset import MyDataSet
import numpy as np
import torch.nn as nn
import torch
from models.lstm import lstm
from torch.autograd import Variable
import matplotlib.pyplot as plt



# STEP.1:导入原始数据集
demo=MyDataSet('C:/Users/Administrator/Desktop/train2(3).csv')

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
train_X = x[:train_size]
train_Y = y[:train_size]
test_X = x[train_size:]
test_Y = y[train_size:]

# STEP.6:调整数据形状
train_X = np.reshape(train_X, (-1, 1, 10))
test_X = np.reshape(test_X, (-1, 1, 10))

train_Y=np.reshape(train_Y, (-1,1, 1))
#test_Y = np.reshape(test_X, (1, 218, 1))

print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
#print(train_X[0], train_Y[0], test_X[0], test_Y[0])

s,b,h=torch.from_numpy(train_X).size()


# STEP.7:建立损失函数和优化器
model = lstm()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# STEP.8:训练
for e in range(200):
    var_x =Variable(torch.from_numpy(train_X)).float()
    var_y = Variable(torch.from_numpy(train_Y)).type(torch.float32)

    out = model(var_x)
    loss = criterion(out, var_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if (e + 1) % 10 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

# STEP.9:测试
model = model.eval()
pred_x = torch.from_numpy(test_X).type(torch.float32)
pred_x = Variable(pred_x)
pred_test = model(pred_x)
print(len(pred_test))
pred_test = pred_test.view(-1).data.numpy()
print(len(pred_test))

pred_y = torch.from_numpy(test_Y).type(torch.float32)
pred_out = pred_y.view(-1).data.numpy()
plt.plot(pred_test, 'r', label ='prediction')
plt.plot(pred_out, 'b', label ='real')
plt.legend(loc='best')
plt.show()
