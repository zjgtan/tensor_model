import sys
sys.path.append(".")
import tensorflow as tf
from tensorflow import keras
from model.feature_column import *


class DNN(keras.Model):
    def __init__(self, feature_map, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        self.feature_map = feature_map # 包含了模型用到的所有特征列信息
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.embedding_layer = self.create_embedding_layer(vocab_size)
        self.mlp = self.get_mlp_block(hidden_units, ["relu"] * (len(hidden_units) - 1) + ["linear"])


    def create_embedding_layer(self, vocab_size):
        embedding_layer = keras.layers.Embedding(input_dim=vocab_size,
                                                    output_dim=self.embedding_dim)
        return embedding_layer

    def get_mlp_block(self, hidden_units, activations):
        mlp = tf.keras.Sequential()

        for idx in range(len(hidden_units)):
            mlp.add(keras.layers.Dense(hidden_units[idx], activation=activations[idx]))

        return mlp

    def embedding_lookup(self, inputs):
        embedding_dict = {}
        for name, feature_column in self.feature_map.items():
            embedding = self.embedding_layer(inputs[name])
            if isinstance(feature_column, VarLenColumn):
                embedding = tf.reduce_sum(embedding, axis=2).to_tensor()
            embedding_dict[name] = tf.squeeze(embedding, axis=1)

        return embedding_dict

    def concat_embedding(self, embedding_dict):
        embedding_list = []
        for name, feature_columnn in self.feature_map.items():
            embedding_list.append(embedding_dict[name])

        concated_embedding = tf.concat(embedding_list, axis=-1)

        return concated_embedding
    
    def call(self, inputs):
        embedding_dict = self.embedding_lookup(inputs)
        concated_embedding = self.concat_embedding(embedding_dict)
        logits = self.mlp(concated_embedding)
        return [logits]

