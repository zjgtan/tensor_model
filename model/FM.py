import tensorflow as tf
from tensorflow import keras

class FM(keras.Model):
    def __init__(self, emb_size, vocab_size):
        super().__init__()
        self.order2_embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_size)
        self.order1_embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=1)

    
    def call(self, inputs):
        order2_embeddings = self.order2_embedding_layer(inputs) # [B, N, emb_size]
        sum_square_emb = tf.square(tf.reduce_sum(order2_embeddings, axis=1, keepdims=True))
        square_sum_emb = tf.reduce_sum(tf.square(order2_embeddings), axis=1, keepdims=True)
        order2_logits = 0.5 * (tf.reduce_sum(sum_square_emb, axis=-1) - tf.reduce_sum(square_sum_emb, axis=-1))

        order1_embeddings = self.order1_embedding_layer(inputs) # [B, N, 1]
        order1_logits = tf.reduce_sum(order1_embeddings, axis=1)

        logits = order1_logits + order2_logits

        return logits


if __name__ == "__main__":
    model = FM(3, 10)
    inputs = tf.constant([[1, 1, 1, 1], [2, 2, 2, 2]])
    model(inputs)



        