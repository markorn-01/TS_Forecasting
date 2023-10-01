# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Conv1D

# class TransformerEncoder(Model):
#     def __init__(self, num_features,
#                        num_heads=2,
#                        ff_dim=32,
#                        dropout=0.3,
#                        num_transformer_layers=2):
#         super(TransformerEncoder, self).__init__()
#         self.num_features = num_features
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.dropout = dropout
#         self.num_transformer_layers = num_transformer_layers
#         self.key_dim = self.num_features // self.num_heads
        
#         # Multi-head self-attention layer
#         self.attn_layer = MultiHeadAttention(num_heads=self.num_heads, 
#                                              key_dim=self.key_dim)
#         self.dropout1 = Dropout(self.dropout)
#         self.layernorm = LayerNormalization(epsilon=1e-6)
        
#         # Feed-forward neural network layer(s)
#         self.conv_layers = []
#         for _ in range(self.num_transformer_layers - 1):
#             self.conv_layers.append(Conv1D(self.ff_dim, kernel_size=1, activation='relu'))
#         self.dropout2 = Dropout(self.dropout)
        
#     def call(self, inputs):
#         attn_layer = self.attn_layer(inputs, inputs)
#         attn_layer = self.dropout1(attn_layer)
#         attn_out = self.layernorm(inputs + attn_layer)
        
#         ff_layer = attn_out
#         for conv_layer in self.conv_layers:
#             ff_layer = conv_layer(ff_layer)
#         ff_layer = self.dropout2(ff_layer)
#         return attn_out, ff_layer

# class TransformerDecoder(Model):
#     def __init__(self, num_features):
#         super(TransformerDecoder, self).__init__()
        
#         self.output_layer1 = Dense(num_features, activation='relu')
#         self.layernorm = LayerNormalization(epsilon=1e-6)
#         self.output_layer2 = Dense(num_features, activation='linear')
        
#     def call(self, encoder_output):
#         attn_out, ff_layer = encoder_output
#         outputs = self.output_layer1(ff_layer)
#         outputs = self.layernorm(attn_out + outputs) 
#         outputs = self.output_layer2(outputs)
#         return outputs

# class Transformer(Model):
#     def __init__(self, num_features,
#                        num_heads=2,
#                        ff_dim=32,
#                        dropout=0.3,
#                        num_transformer_layers=2):
#         super(Transformer, self).__init__()
#         self.encoder = TransformerEncoder(num_features, num_heads, ff_dim, dropout, num_transformer_layers)
#         self.decoder = TransformerDecoder(num_features)
        
#     def call(self, inputs):
#         encoder_output = self.encoder(inputs)
#         outputs = self.decoder(encoder_output)
#         return outputs

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Conv1D

class Transformer(Model):
    def __init__(self, num_features,
                       num_heads=8,
                       ff_dim=32,
                       dropout=0.3,
                       num_transformer_layers=2):
        super(Transformer, self).__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_transformer_layers = num_transformer_layers
        self.key_dim = self.num_features // self.num_heads
        
        # Multi-head self-attention layer
        self.attn_layer = MultiHeadAttention(num_heads=self.num_heads, 
                                             key_dim=self.key_dim)
        self.dropout1 = Dropout(self.dropout)
        self.layernorm1 = LayerNormalization(epsilon=1e-6, trainable=False)
        
        # Feed-forward neural network layer(s)
        self.conv_layers = []
        for _ in range(self.num_transformer_layers - 1):
            self.conv_layers.append(Conv1D(self.ff_dim, kernel_size=1, activation='relu'))
        self.dropout2 = Dropout(self.dropout)
        self.layernorm2 = LayerNormalization(epsilon=1e-6, trainable=False)
        
        # Output dense layer
        self.output_layer1 = Dense(self.num_features, activation='relu')
        self.output_layer2 = Dense(self.num_features, activation='linear')
        
    def call(self, inputs):
        attn_layer = self.attn_layer(inputs, inputs)
        attn_layer = self.dropout1(attn_layer)
        attn_out = self.layernorm1(inputs + attn_layer)
        ff_layer = attn_out
        for conv_layer in self.conv_layers:
            ff_layer = conv_layer(ff_layer)
        ff_layer = self.dropout2(ff_layer)
        ff_layer = self.output_layer1(ff_layer)
        ff_out = self.layernorm2(attn_out + ff_layer)
        ff_out = attn_out + ff_layer
        outputs = self.output_layer2(ff_out)
        return outputs