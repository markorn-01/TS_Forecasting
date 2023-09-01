import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Conv1D

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
        self.attn_layer = MultiHeadAttention(num_heads=self.num_heads, 
                                             key_dim=self.key_dim)
        self.dropout1 = Dropout(self.dropout)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        
        # Feed-forward neural network layer(s)
        self.conv_layers = []
        for _ in range(self.num_transformer_layers - 1):
            self.conv_layers.append(Conv1D(self.ff_dim, kernel_size=1, activation='relu'))
        self.dropout2 = Dropout(self.dropout)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Output dense layer
        self.output_layer1 = Dense(self.num_features, activation='relu')
        self.output_layer2 = Dense(self.num_features, activation='linear')
        
    def call(self, inputs):
        attn_layer = self.attn_layer(inputs, inputs)
        attn_layer = self.dropout1(attn_layer)
        attn_out = self.layernorm1(inputs + attn_layer)
        attn_out = inputs + attn_layer
        ff_layer = attn_out
        for conv_layer in self.conv_layers:
            ff_layer = conv_layer(ff_layer)
        ff_layer = self.dropout2(ff_layer)
        ff_layer = self.output_layer1(ff_layer)
        ff_out = self.layernorm2(attn_out + ff_layer)
        ff_out = attn_out + ff_layer
        outputs = self.output_layer2(ff_out)
        return outputs
