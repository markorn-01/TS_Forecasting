import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dense
class CNN(Model):
    def __init__(self, conv_width):
        super(CNN, self).__init__()
        self.conv_layer = Conv1D(filters=32,
                                    kernel_size=(conv_width,),
                                    activation='relu')
        # self.max_pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.dense_layer1 = Dense(units=32, activation='relu')
        self.dense_layer2 = Dense(units=1)
        
    def call(self, inputs):
        x = self.conv_layer(inputs)
        # x = self.max_pooling_layer(x)
        x = self.dense_layer1(x)
        output = self.dense_layer2(x)
        return output