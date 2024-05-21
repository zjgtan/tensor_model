import tensorflow as tf
from tensorflow import keras

class AutoInt(keras.Model):
    def __init__(self, vocab_size, emb_size, self_attention_linear_projection_num_units, self_attention_num_heads):
        super().__init__()

        self.self_attention_linear_projection_num_units = self_attention_linear_projection_num_units
        self.embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_size)

        self.Q = keras.layers.Dense(self_attention_linear_projection_num_units, activation="relu")
        self.K = keras.layers.Dense(self_attention_linear_projection_num_units, activation="relu")
        self.V = keras.layers.Dense(self_attention_linear_projection_num_units, activation="relu")

        self.self_attention_num_heads = self_attention_num_heads

        self.pred_layer = keras.layers.Dense(1)


    def multi_head_self_attention(self, embedding):

        Q_ = tf.concat(tf.split(self.Q(embedding), self.self_attention_num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(self.K(embedding), self.self_attention_num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(self.V(embedding), self.self_attention_num_heads, axis=2), axis=0)

        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)
        # Activation
        weights = tf.nn.softmax(weights)
        # Weighted sum
        outputs = tf.matmul(weights, V_)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.self_attention_num_heads, axis=0), axis=2)

        outputs = tf.nn.relu(outputs)

        return outputs

    def call(self, inputs, fix_len_feature_columns, var_len_feature_columns):

        field_size = len(fix_len_feature_columns) + len(var_len_feature_columns)

        embedding_list = []
        # 向量化
        for feature_column in fix_len_feature_columns:
            embedding_list.append(self.embedding_layer(inputs[feature_column]))

        for feature_column in var_len_feature_columns:
            embedding = tf.reduce_mean(self.embedding_layer(inputs[feature_column]), axis=1).to_tensor()
            embedding_list.append(embedding)


        embedding = tf.concat(embedding_list, axis=1)

        # transformer
        self_attention_embedding_output = self.multi_head_self_attention(embedding)
        merged_output = tf.reshape(self_attention_embedding_output, shape=[-1, field_size * self.self_attention_linear_projection_num_units])

        # output 
        logits = self.pred_layer(merged_output)


        return logits

