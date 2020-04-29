import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
class op():
    def __init__(self, df):
        self.df = df
        self.df['纬度'] = self.df['纬度'].astype(float)
        self.df['经度'] = self.df['经度'].astype(float)
    def divide_into_XY(self, look_back=10, latitude=True):
        dataX, dataY = [], []
        if latitude:
            for i in range(len(self.df) - look_back):
                a = self.df['纬度'][i:(i + look_back)]
                dataX.append(a)
                dataY.append(self.df['纬度'][i + look_back])
        else:
            for i in range(len(self.df) - look_back):
                a = self.df['经度'][i:(i + look_back)]
                dataX.append(a)
                dataY.append(self.df['经度'][i + look_back])

        return (dataX, dataY)

    def preprocessor(self, input_data = None):
        scaler_data = MinMaxScaler()
        data = scaler_data.fit_transform(input_data)
        return data

    #def divide_into_TV(self,spilt=1):

    def test(self):
        print(self.df['经度'])