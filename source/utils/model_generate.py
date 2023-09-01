import tensorflow as tf

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                  optimizer=tf.keras.optimizers.legacy.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.MeanSquaredError()])

    model.fit(window.train, 
                        epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return model
