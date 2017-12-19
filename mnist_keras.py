import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

class MnistPrediction:
    
    def __init__(self, batch_size=100, epochs=1):
        self.batch_size = batch_size
        self.output_classes = 10
        self.epochs = epochs
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = self.x_train.reshape(60000, 784).astype('float32') / 255
        self.x_test = self.x_test.reshape(10000, 784).astype('float32') / 255
        self.y_train = keras.utils.to_categorical(self.y_train, self.output_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.output_classes)

    def run(self):
        self._evalate(self._fit(self._create_model()))
        
    def _create_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.output_classes, activation='softmax'))
        #model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        return model

    def _fit(self, model):
        self.history = model.fit(self.x_train, self.y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=(self.x_test, self.y_test))
        return model
        
    def _evalate(self, model):
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss: {0}, Test accuracy: {1}'.format(score[0], score[1]))

if __name__ == "__main__":
    MnistPrediction(batch_size=100, epochs=1).run()
    #MnistPrediction(batch_size=100, epochs=2).predict()
    #MnistPrediction(batch_size=100, epochs=3).predict()
    #MnistPrediction(batch_size=100, epochs=4).predict()
    #MnistPrediction(batch_size=100, epochs=5).predict()
