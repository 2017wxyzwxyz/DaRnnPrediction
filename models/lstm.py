from torch import nn

class lstm(nn.Module):
    def __init__(self,input_size=10, hidden_size=20, ouput_size=1,num_layer=2):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size , num_layer)
        self.layer2 = nn.Linear(hidden_size, ouput_size)

    def forward(self, x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b, h)
        x = self.layer2(x)
        x = x.view(s, b,-1)
        return x