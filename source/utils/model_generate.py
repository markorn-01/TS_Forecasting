import tensorflow as tf


def compile_and_fit(model, window, patience=2, 
                    loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.legacy.Adam(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError(),
                             tf.keras.metrics.RootMeanSquaredError(),
                             tf.keras.metrics.MeanAbsolutePercentageError()],
                    MAX_EPOCHS=50):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    model.fit(window.train, 
                epochs=MAX_EPOCHS,
                validation_data=window.val,
                callbacks=[early_stopping])
    return model
