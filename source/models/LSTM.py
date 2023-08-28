import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel(Model):
    def __init__(self, lstm_units):
        super(LSTMModel, self).__init__()
        self.lstm_layer = LSTM(lstm_units, return_sequences=True)
        self.dense_layer = Dense(units=1)
        
    def call(self, inputs):
        lstm_output = self.lstm_layer(inputs)
        predictions = self.dense_layer(lstm_output)
        return predictions