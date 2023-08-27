# import keras
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# class CNN:
#     def __init__(self, window):
#         self.num_features = window.train_df.shape[-1]
#         self.input_shape = (window.input_width, self.num_features)
#         self.num_classes = 1
#         self.model = self.call()

#     def call(self):
#         self.model = Sequential()

#         self.model.add(Conv1D(32, 3, activation='relu', input_shape=self.input_shape))
#         self.model.add(MaxPooling1D(2))

#         self.model.add(Conv1D(64, 3, activation='relu'))
#         self.model.add(MaxPooling1D(2))

#         self.model.add(Flatten())
#         self.model.add(Dense(128, activation='relu'))
#         self.model.add(Dense(self.num_classes, activation='softmax'))
#         return self.model

import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, conv_width):
        super(CNN, self).__init__()
        self.conv_layer = tf.keras.layers.Conv1D(filters=32,
                                                 kernel_size=(conv_width,),
                                                 activation='relu')
        # self.max_pooling_layer = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.dense_layer1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense_layer2 = tf.keras.layers.Dense(units=1)
        
    def call(self, inputs):
        x = self.conv_layer(inputs)
        # x = self.max_pooling_layer(x)
        x = self.dense_layer1(x)
        output = self.dense_layer2(x)
        return output