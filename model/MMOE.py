import sys
sys.path.append(".")
import tensorflow as tf
from tensorflow import keras
from model.feature_column import *


class MMOE(keras.Model):
    def __init__(self, feature_map, vocab_size, embedding_dim, task_num, num_experts, expert_hidden_units, gate_hidden_units, tower_hidden_units):
        super().__init__()
        self.feature_map = feature_map # 包含了模型用到的所有特征列信息
        self.embedding_dim = embedding_dim
        self.task_num = task_num
        self.num_experts = num_experts
        self.expert_hidden_units = expert_hidden_units
        self.gate_hidden_units = gate_hidden_units
        self.tower_hidden_units = tower_hidden_units

        self.embedding_layer = self.create_embedding_layer(vocab_size)

        self.experts = [self.get_mlp_block(expert_hidden_units, ["relu"] * len(expert_hidden_units)) for _ in range(num_experts)]

        self.gates = [self.get_mlp_block(gate_hidden_units, ["relu"] * (len(gate_hidden_units) - 1) + ["softmax"]) for _ in range(task_num)]

        self.towers = [self.get_mlp_block(tower_hidden_units, ["relu"] * (len(tower_hidden_units) - 1) + ["linear"]) for _ in range(task_num)]


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

        output_experts = [self.experts[idx](concated_embedding) for idx in range(self.num_experts)]
        output_gates = [self.gates[idx](concated_embedding) for idx in range(self.task_num)]

        output_experts = tf.concat([e[:, tf.newaxis, :] for e in output_experts], axis=1) # [bs, n_expert, dim]

        multi_task_logits = []
        for idx in range(self.task_num):
            g = tf.expand_dims(output_gates[idx], axis=-1) # [bs, n_expert, 1]
            merged_expert = tf.squeeze(tf.matmul(output_experts, g, transpose_a=True), axis=-1)
            tower_logits = self.towers[idx](merged_expert)
            multi_task_logits.append(tower_logits)

        return multi_task_logits


if __name__ == "__main__":
    feature_map = {"mid": SparseColumn("mid", "user"),
                "iid": SparseColumn("iid", "item"),
                "scene": SparseColumn("scene", "domain")}

    net = MMOE(feature_map, 10, 5, 2, 3, [10, 10], [10, 3], [10, 1])

    batch = {"mid": tf.constant([[1], [2]]),
             "iid": tf.constant([[1], [2]]),
             "scene": tf.constant([[1], [2]])}

    print(net(batch))







