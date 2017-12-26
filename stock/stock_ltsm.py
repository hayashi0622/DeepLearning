# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

class Prediction:

    def __init__(self, length_of_sequences = 10, hidden_neurons = 300):
        self.length_of_sequences = length_of_sequences
        self.in_out_neurons = 1
        self.hidden_neurons = hidden_neurons

    def load_data(self, data, n_prev=10):
        X, Y = [], []
        for i in range(len(data) - n_prev):
            X.append(data.iloc[i:(i+n_prev)].as_matrix())
            Y.append(data.iloc[i+n_prev].as_matrix())
        retX = np.array(X)
        retY = np.array(Y)
        return retX, retY

    def create_model(self):
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, \
                       batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), \
                       return_sequences=False))
        model.add(Dense(self.in_out_neurons))
        model.add(Activation("linear"))
        model.compile(optimizer="adam",
                      loss="mape")
        return model

    def train(self, X_train, y_train, epochs=100):
        model = self.create_model()
        model.fit(X_train, y_train, batch_size=10, epochs=epochs)
        return model

def _run(file_prefix, length_of_sequences, hidden_neurons, epochs):
    p = Prediction(length_of_sequences, hidden_neurons)
    data = None
    for year in range(2007, 2018):
        data_ = pd.read_csv('csv/{0}_{1}.csv'.format(file_prefix, year))
        data = data_ if (data is None) else pd.concat([data, data_])
    data.columns = ['date', 'open', 'high', 'low', 'close']
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data['close'] = preprocessing.scale(data['close'])
    data = data.sort_values(by='date')
    data = data.reset_index(drop=True)
    data = data.loc[:, ['date', 'close']]
    split_pos = int(len(data) * 0.9)
    x_train, y_train = p.load_data(data[['close']].iloc[0:split_pos], p.length_of_sequences)
    x_test,  y_test  = p.load_data(data[['close']].iloc[split_pos:], p.length_of_sequences)
    dates = data[['date']].iloc[split_pos + p.length_of_sequences:]
    model = p.train(x_train, y_train, epochs)
    predicted = model.predict(x_test)
    result = pd.DataFrame(predicted, index=dates['date'])
    result.columns = ['predict']
    result['actual'] = y_test
    plot = result.plot(
            title='Nikkei Stock Average <LoS={0},HN={1},EP={2}>'.format(
                    length_of_sequences,
                    hidden_neurons,
                    epochs),
            figsize=(8, 5),
            grid=True)
    fig = plot.get_figure()
    fig.savefig('fig/{0}_los{1:02d}_hn{2:03d}_ep{3:03d}.png'.format(
            file_prefix, length_of_sequences, hidden_neurons, epochs))

if __name__ == "__main__":
    _run(file_prefix = 'indices_I101_1d', length_of_sequences = 10, hidden_neurons = 100, epochs = 1)
    '''
    for s in [10, 20, 30]:
        for n in [100, 300, 500]:
            for e in [1, 10, 100]:
                _run(file_prefix = 'indices_I101_1d',
                     length_of_sequences = s,
                     hidden_neurons = n,
                     epochs = e)
                break
    '''
