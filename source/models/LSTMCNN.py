import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.models import Model
from models.LSTM import LSTMModel
from models.CNN import CNN
class LSTMCNN(Model):
    def __init__(self, lstm_units, conv_width):
        super(LSTMCNN, self).__init__()
        self.lstm_layer = LSTMModel(lstm_units)
        self.conv_layer = CNN(conv_width=conv_width)
        self.dense_layer1 = Dense(units=32, activation='relu')
        self.dense_layer2 = Dense(units=1)

    def call(self, inputs):
        lstm_output = self.lstm_layer(inputs)
        cnn_output = self.conv_layer(inputs)
        combined_output = Concatenate(axis=-1)([lstm_output, cnn_output])
        x = self.dense_layer1(combined_output)
        output = self.dense_layer2(x)
        return output