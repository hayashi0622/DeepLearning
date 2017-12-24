# -*- coding: utf-8 -*-
import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

class Prediction:

    def __init__(self):
        self.length_of_sequences = 10
        self.in_out_neurons = 1
        self.hidden_neurons = 300

    def load_data(self, data, n_prev=10):
        X, Y = [], []
        for i in range(len(data) - n_prev):
            X.append(data.iloc[i:(i+n_prev)].as_matrix())
            Y.append(data.iloc[i+n_prev].as_matrix())
        retX = numpy.array(X)
        retY = numpy.array(Y)
        return retX, retY

    def create_model(self):
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, \
                       batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), \
                       return_sequences=False))
        model.add(Dense(self.in_out_neurons))
        model.add(Activation("linear"))
        model.compile(loss="mape", optimizer="adam")
        return model

    def train(self, X_train, y_train, epochs=100):
        model = self.create_model()
        model.fit(X_train, y_train, batch_size=100, epochs=epochs)
        return model

def _run(file_prefix, epochs):
    p = Prediction()
    # データ準備
    data = None
    for year in range(2007, 2017):
        data_ = pandas.read_csv('csv/' + file_prefix + '_' + str(year) +  '.csv')
        data = data_ if (data is None) else pandas.concat([data, data_])
    data.columns = ['date', 'open', 'high', 'low', 'close']
    data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
    # 終値のデータを標準化
    data['close'] = preprocessing.scale(data['close'])
    data = data.sort_values(by='date')
    data = data.reset_index(drop=True)
    data = data.loc[:, ['date', 'close']]
    # 2割をテストデータへ
    split_pos = int(len(data) * 0.8)
    x_train, y_train = p.load_data(data[['close']].iloc[0:split_pos], p.length_of_sequences)
    x_test,  y_test  = p.load_data(data[['close']].iloc[split_pos:], p.length_of_sequences)
    model = p.train(x_train, y_train, epochs)
    predicted = model.predict(x_test)
    result = pandas.DataFrame(predicted)
    result.columns = ['predict']
    result['actual'] = y_test
    plot = result.plot(title='Stock prices prediction (epoch: {0})'.format(epochs))
    fig = plot.get_figure()
    fig.savefig('epoch_{0:03d}.png'.format(epochs))

if __name__ == "__main__":
    _run('indices_I101_1d', 1)
    _run('indices_I101_1d', 10)
    _run('indices_I101_1d', 50)
    _run('indices_I101_1d', 100)
    _run('indices_I101_1d', 200)
