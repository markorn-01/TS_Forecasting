import tensorflow as tf


def compile_and_fit(model, window, patience=2, 
                    loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.legacy.Adam(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError(),
                             tf.keras.metrics.MeanSquaredError()],
                    MAX_EPOCHS=20,
                    checkpoint_filepath=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    model.fit(window.train, 
                epochs=MAX_EPOCHS,
                validation_data=window.val,
                callbacks=[early_stopping, model_checkpoint_callback])
    return model
