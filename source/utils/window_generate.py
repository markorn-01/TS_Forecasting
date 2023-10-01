import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df, val_df, test_df, label_columns):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] \
                    for name in self.label_columns],
                axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot_inputs(self, inputs):
        plt.plot(self.input_indices, inputs, label='Inputs', marker='.', zorder=-10)

    def plot_labels(self, labels):
        plt.scatter(self.label_indices, labels, edgecolors='k', label='Labels', c='#2ca02c', s=64)

    def plot_predictions(self, predictions):
        plt.scatter(self.label_indices, predictions, marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)

    def plot(self, model=None, plot_col=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            self.plot_inputs(inputs[n, :, plot_col_index])
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue
            self.plot_labels(labels[n, :, label_col_index])
            if model is not None:
                predictions = model(inputs)
                self.plot_predictions(predictions[n, :, label_col_index])
            if n == 0:
                plt.legend()
        plt.xlabel('Time (day)')
        plt.show()


    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        if not getattr(self, '_example', None):
            # No example batch was found, so get one from the `.train` dataset
            # And cache it for next time
            self._example = next(iter(self.train))
        return self._example