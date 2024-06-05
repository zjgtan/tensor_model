import tensorflow as tf
from tensorflow import keras
from feature_column import *


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
    def __init__(self, gate_hidden_dim, embedding_dim, feature_group_num_dict):
        super().__init__()
        self.feature_group_num_dict = feature_group_num_dict
        self.gate_unit = GateNU(gate_hidden_dim, embedding_dim * (feature_group_num_dict["general"] + feature_group_num_dict["domain"]))

    def call(self, domain_embedding, general_embedding):
        input_embedding = tf.concat([domain_embedding, tf.stop_gradient(general_embedding)], axis=-1)
        gate = self.gate_unit(input_embedding)
        output = tf.tile(gate, [1, len(self.feature_map)]) * general_embedding
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
    def __init__(self, feature_map, embedding_dim, task_num, dnn_hidden_units, gate_hidden_dim):
        super().__init__()
        self.feature_map = feature_map # 包含了模型用到的所有特征列信息
        self.embedding_dim = embedding_dim
        self.task_num = task_num
        self.embedding_layer_dict = self.create_embedding_layer(feature_map)

        self.epnet = EPNet(gate_hidden_dim, embedding_dim, self.feature_map)
        self.ppnet = PPNet(gate_hidden_dim, dnn_hidden_units, task_num)

    def create_embedding_layer(self, feature_map):
        embedding_layer_dict = {}
        for name, feature_column in feature_map.items():
            if isinstance(feature_column, SparseColumn) or isinstance(feature_column, VarLenColumn):
                embedding_layer = keras.layers.Embedding(input_dim=feature_column.vocabulary_size,
                                                         output_dim=self.embedding_dim)
                embedding_layer_dict[name] = embedding_layer

        return embedding_layer_dict

    def embedding_lookup(self, inputs):
        embedding_dict = {}
        for name, feature_column in self.feature_map.items():
            embedding = self.embedding_layer_dict[name](inputs[name])
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


if __name__ == "__main__":
    feature_map = {"mid": SparseColumn("mid", 10, 4, "user"), 
                       "iid": SparseColumn("iid", 10, 4, "item"),
                       "src": SparseColumn("src", 10, 4, "domain")}

    inputs = {"mid": tf.constant([[1], [2]]),
              "iid": tf.constant([[3], [4]]),
              "src": tf.constant([[5], [6]])}
    

    model = PEPNet(feature_map=feature_map, embedding_dim=4, task_num=2, dnn_hidden_units=[64, 64, 64], gate_hidden_dim=64)

    output = model(inputs)

    print(output)