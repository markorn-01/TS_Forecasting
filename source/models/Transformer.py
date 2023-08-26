import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization

class Transformer:
    def __init__(self, window,
                       num_heads = 2,
                       ff_dim = 32,
                       dropout = 0.3,
                       num_transformer_layers = 2):
        self.num_features = window.train_df.shape[-1]
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_transformer_layers = num_transformer_layers
        self.inputs = Input(shape=(window.input_width, self.num_features))
        self.key_dim = self.inputs.shape[1] // self.num_heads
        self.model = self.call()
        
    # Define a custom Transformer model
    def call(self):
        # Multi-head self-attention layer
        attn_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)(self.inputs, self.inputs)
        attn_layer = Dropout(self.dropout)(attn_layer)
        attn_out = LayerNormalization(epsilon=1e-6)(self.inputs + attn_layer)
        
        # Feed-forward neural network layer
        ff_layer = attn_out
        for _ in range(self.num_transformer_layers-1):
            ff_layer = Dense(self.ff_dim, activation='relu')(ff_layer)
        ff_layer = Dropout(self.dropout)(ff_layer)
        ff_layer = Dense(self.inputs.shape[-1])(ff_layer)
        ff_out = LayerNormalization(epsilon=1e-6)(attn_out + ff_layer)
        outputs = Dense(self.inputs.shape[-1], activation='softmax')(ff_out)
        model = Model(self.inputs, outputs)
        return model


# # Compile and train the model
# transformer.compile(optimizer='adam', loss='mse')
# transformer.fit(window.train, epochs=num_epochs, validation_data=window.val)

# # Evaluate and plot predictions
# evaluation_results = transformer.evaluate(window.test)
# window.plot(model=transformer, plot_col='your_plot_column_name')

