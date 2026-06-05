"""
Model Architecture for Univariate Transformer
==============================================
Contains transformer encoder, model building functions, and learning rate schedules.
"""

import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from .transformer_architechtures.model_architechture_informer import build_informer_model
from .transformer_architechtures.model_architechture_log_transformer import build_log_transformer_model
from .transformer_architechtures.model_architechture_LSTNet import build_lstnet_model
from .transformer_architechtures.model_architechture_base_tranformer import build_base_model 




def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function = "tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding = False):
    #return build_informer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function, dropout, mlp_dropout, n_pred, pos_encoding)
    #return build_lstnet_model(input_shape, n_filters=head_size*num_heads, kernel_size=6, rnn_units=ff_dim, skip_units=ff_dim//2, skip=7, n_pred=n_pred, dropout=dropout)
    return build_base_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function, dropout, mlp_dropout, n_pred, pos_encoding)



from src.core.layers import CustomCosineDecay, PositionalEncoding, RevIN

