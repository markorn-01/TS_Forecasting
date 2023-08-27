import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization

# class Transformer1:
#     def __init__(self, window,
#                        num_heads = 2,
#                        ff_dim = 32,
#                        dropout = 0.3,
#                        num_transformer_layers = 2):
#         self.num_features = window.train_df.shape[-1]
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.dropout = dropout
#         self.num_transformer_layers = num_transformer_layers
#         self.inputs = Input(shape=(window.input_width, self.num_features))
#         self.key_dim = self.inputs.shape[1] // self.num_heads
#         self.model = self.call()
        
#     # Define a custom Transformer model
#     def call(self):
#         # Multi-head self-attention layer
#         attn_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)(self.inputs, self.inputs)
#         attn_layer = Dropout(self.dropout)(attn_layer)
#         attn_out = LayerNormalization(epsilon=1e-6)(self.inputs + attn_layer)
        
#         # Feed-forward neural network layer
#         ff_layer = attn_out
#         for _ in range(self.num_transformer_layers-1):
#             ff_layer = Dense(self.ff_dim, activation='relu')(ff_layer)
#         print(ff_layer.shape)
#         ff_layer = Dropout(self.dropout)(ff_layer)
#         ff_layer = Dense(self.inputs.shape[-1])(ff_layer)
#         ff_out = LayerNormalization(epsilon=1e-6)(attn_out + ff_layer)
#         outputs = Dense(self.inputs.shape[-1], activation='softmax')(ff_out)
#         print("outputs: ",outputs.shape)
#         model = Model(self.inputs, outputs)
#         return model


# # # Compile and train the model
# # transformer.compile(optimizer='adam', loss='mse')
# # transformer.fit(window.train, epochs=num_epochs, validation_data=window.val)

# # # Evaluate and plot predictions
# # evaluation_results = transformer.evaluate(window.test)
# # window.plot(model=transformer, plot_col='your_plot_column_name')

# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization
# from tensorflow.keras.models import Model

class Transformer(Model):
    def __init__(self, window,
                       num_heads=2,
                       ff_dim=32,
                       dropout=0.3,
                       num_transformer_layers=2):
        super(Transformer, self).__init__()
        self.window = window
        self.num_features = window.train_df.shape[-1]
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_transformer_layers = num_transformer_layers
        self.key_dim = self.num_features // self.num_heads
        
        # Multi-head self-attention layer
        self.attn_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.dropout1 = Dropout(self.dropout)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        
        # Feed-forward neural network layer(s)
        # self.dense_layers = [Dense(self.ff_dim, activation='relu') for _ in range(self.num_transformer_layers-1)]
        self.dropout2 = Dropout(self.dropout)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Output dense layer
        self.output_layer1 = Dense(self.num_features)
        self.output_layer2 = Dense(self.num_features, activation='softmax')
        
    def call(self, inputs):
        inputs = Input(shape=(self.window.input_width, self.num_features))
        attn_layer = self.attn_layer(inputs, inputs)
        attn_layer = self.dropout1(attn_layer)
        attn_out = self.layernorm1(inputs + attn_layer)
        
        ff_layer = attn_out
        # for dense_layer in self.dense_layers:
        #     ff_layer = dense_layer(ff_layer)
        for _ in range(self.num_transformer_layers-1):
            ff_layer = Dense(self.ff_dim, activation='relu')(ff_layer)
        # print(ff_layer.shape)
        ff_layer = self.dropout2(ff_layer)
        ff_layer = self.output_layer1(ff_layer)
        ff_out = self.layernorm2(attn_out + ff_layer)
        # print("ff_out: ",ff_out.shape)
        outputs = self.output_layer2(ff_out)
        # print("outputs: ",outputs.shape)
        return outputs
