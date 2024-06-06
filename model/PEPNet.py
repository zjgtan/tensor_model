import tensorflow as tf
from tensorflow import keras
from model.feature_column import *


"""
目前线上的结构

ep: 
    beta0, src_id, 业务类型，推广目的, unit_id

pp:
    一级行业、二级行业、mid, 偏好标签，年龄、性别，手机型号，user_id, unit_id, plan_id, 游戏app，

"""

class GateNU(keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        layers = [
            keras.layers.Dense(hidden_dim, "relu"),
            keras.layers.Dense(output_dim, "sigmoid")
        ]
        
        self.gate_layer = keras.Sequential(layers)
        
    def call(self, inputs):
        return self.gate_layer(inputs) * 2


class EPNet(keras.Model):
    def __init__(self, gate_hidden_dim, embedding_dim, feature_map):
        super().__init__()
        self.feature_map = feature_map
        self.slot_num = sum([1 for feat, col in self.feature_map.items() if col.group in ["general", "user", "item"]])
        self.gate_unit = GateNU(gate_hidden_dim, embedding_dim * slot_num)

    def call(self, domain_embedding, general_embedding):
        input_embedding = tf.concat([domain_embedding, tf.stop_gradient(general_embedding)], axis=-1)
        gate = self.gate_unit(input_embedding)
        output = gate * general_embedding
        return output


class PPNet(keras.Model):
    def __init__(self, gate_hidden_dim, dnn_hidden_units, task_num):
        super().__init__()

        self.gate_units = [GateNU(gate_hidden_dim, dnn_hidden_units[i] * task_num) for i in range(len(dnn_hidden_units))]

        self.task_towers = []
        for task_idx in range(task_num):
            tower_layer_list = []
            for hidden_unit in dnn_hidden_units:
                tower_layer_list.append(keras.layers.Dense(hidden_unit, activation="relu"))
            tower_layer_list.append(keras.layers.Dense(1))
            
            self.task_towers.append(tower_layer_list)

        self.task_num = task_num
        self.dnn_hidden_units = dnn_hidden_units

    def call(self, input_embedding, user_group_embedding, item_group_embedding):
        ppnet_input_embedding = tf.concat([user_group_embedding, item_group_embedding, tf.stop_gradient(input_embedding)], axis=-1)
        gates = [tf.split(self.gate_units[i](ppnet_input_embedding), num_or_size_splits=self.task_num, axis=-1) for i in range(len(self.dnn_hidden_units))]

        h = input_embedding
        task_output = []
        for task_idx in range(self.task_num):
            for layer_idx in range(len(self.task_towers[task_idx]) - 1):
                h = self.task_towers[task_idx][layer_idx](h)
                h = h * gates[layer_idx][task_idx]

            out = self.task_towers[task_idx][-1](h)
            task_output.append(out)

        return task_output


class PEPNet(keras.Model):
    def __init__(self, feature_map, vocab_size, embedding_dim, task_num, dnn_hidden_units, gate_hidden_dim):
        super().__init__()
        self.feature_map = feature_map # 包含了模型用到的所有特征列信息
        self.embedding_dim = embedding_dim
        self.task_num = task_num
        self.embedding_layer = self.create_embedding_layer(vocab_size)

        self.epnet = EPNet(gate_hidden_dim, embedding_dim, self.feature_map)
        self.ppnet = PPNet(gate_hidden_dim, dnn_hidden_units, task_num)

    def create_embedding_layer(self, vocab_size):
        embedding_layer = keras.layers.Embedding(input_dim=vocab_size,
                                                    output_dim=self.embedding_dim)
        return embedding_layer

    def embedding_lookup(self, inputs):
        embedding_dict = {}
        for name, feature_column in self.feature_map.items():
            embedding = self.embedding_layer(inputs[name])
            embedding_dict[name] = tf.squeeze(embedding, axis=1)

        return embedding_dict

    def concat_grouped_embedding(self, embedding_dict, group_name="general"):
        embedding_list = []
        for name, feature_columnn in self.feature_map.items():
            if feature_columnn.group_name == group_name or group_name=="general":
                embedding_list.append(embedding_dict[name])

        concated_embedding = tf.concat(embedding_list, axis=-1)

        return concated_embedding

    def call(self, inputs):
        # 查embeddng
        embedding_dict = self.embedding_lookup(inputs)

        general_embedding = self.concat_grouped_embedding(embedding_dict, group_name="general")

        domain_embedding = self.concat_grouped_embedding(embedding_dict, group_name="domain")

        epnet_embedding = self.epnet(domain_embedding, general_embedding)

        # ppnet input
        user_embedding = self.concat_grouped_embedding(embedding_dict, group_name="user")
        item_embedding = self.concat_grouped_embedding(embedding_dict, group_name="item")

        logits = self.ppnet(epnet_embedding, user_embedding, item_embedding)

        return logits
