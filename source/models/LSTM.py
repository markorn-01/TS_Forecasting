import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense
import os
import pickle
from pathlib import Path
class LSTMModel(Model):
    def __init__(self, lstm_units):
        super(LSTMModel, self).__init__()
        self.lstm_layer = LSTM(lstm_units, return_sequences=True)
        self.dense_layer = Dense(units=1)
        
    def call(self, inputs):
        lstm_output = self.lstm_layer(inputs)
        predictions = self.dense_layer(lstm_output)
        return predictions
    
class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        return inputs + delta

class ResidualLSTMModel(Model):
    def __init__(self, lstm_units, num_features):
        super(ResidualLSTMModel, self).__init__()
        self.lstm_layer = LSTM(lstm_units, return_sequences=True)
        self.dense_layer = Dense(num_features, kernel_initializer=tf.initializers.zeros())
        
    def call(self, inputs):
        x = self.lstm_layer(inputs)
        x = self.dense_layer(x)
        return x