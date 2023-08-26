import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

class CNN:
    def __init__(self, window):
        self.num_features = window.train_df.shape[-1]
        self.input_shape = (window.input_width, self.num_features)
        self.num_classes = 1
        self.model = self.call()

    def call(self):
        self.model = Sequential()

        self.model.add(Conv1D(32, 3, activation='relu', input_shape=self.input_shape))
        self.model.add(MaxPooling1D(2))

        self.model.add(Conv1D(64, 3, activation='relu'))
        self.model.add(MaxPooling1D(2))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        return self.model