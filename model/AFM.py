import tensorflow as tf
from tensorflow import keras


class AFM(keras.Model):
    def __init__(self, emb_size, vocab_size, attn_size):
        super().__init__()
        self.order2_embedding_layer = keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_size)
        
        self.attention_net = keras.layers.Dense(attn_size)
        self.projection_layer = keras.layers.Dense(1, use_bias=False)

        self.order2_output_layer = keras.layers.Dense(1)

        
    def call(self, inputs):
        order2_embeddings = self.order2_embedding_layer(inputs)
        print(order2_embeddings)

        element_wise_product_list = []
        for i in range(3):
            for j in range(i+1, 3):
                element_wise_product_list.append(tf.multiply(order2_embeddings[:, i, :], order2_embeddings[:, j, :]))

        print(element_wise_product_list)
        element_wise_product_embeddings = tf.stack(element_wise_product_list, axis=1)
        print("xxxx", element_wise_product_embeddings)

        attn_scores = keras.layers.ReLU()(self.attention_net(tf.reshape(element_wise_product_embeddings, shape=(-1, 3))))
        attn_scores = keras.layers.Softmax()(tf.reshape(self.projection_layer(attn_scores), shape=(-1, 3)))

        weighted_element_wise_product_embeddings = tf.multiply(tf.expand_dims(attn_scores, axis=-1), element_wise_product_embeddings)

        order2_output_logit = self.order2_output_layer(tf.reduce_sum(weighted_element_wise_product_embeddings, axis=1))

        print(order2_output_logit)



        
        
        







if __name__ == "__main__":
    model = AFM(3, 10, 4)
    inputs = tf.constant([[1, 2, 3], [4, 5, 6]])

    model(inputs)
        
        
        
