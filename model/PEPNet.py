import tensorflow as tf
from tensorflow import keras
from feature_column import *

class GateNU(keras.layers)


class PEPNet(keras.Model):
    def __init__(self, feature_map, task_num, task_hidden_units, domain_num):
        super().__init__()
        self.feature_map = feature_map # 包含了模型用到的所有特征列信息
        self.task_num = task_num
        self.embedding_layer_dict = self.create_embedding_layer(feature_map)

        self.task_towers = []
        for _ in range(task_num):
            tower = [
                keras.layers.Dense(task_hidden_units[0], activation='relu'),
                keras.layers.Dense(task_hidden_units[1], activation='relu'),
                keras.layers.Dense(1)
            ]
            self.task_mlp_towers.append(tower)

        self.task_gates = []
        for _ in range(task_num):
            gate = [
                GateNU()
            ]

    def create_embedding_layer(self, feature_map):
        embedding_layer_dict = {}
        for name, feature_column in feature_map.items():
            if isinstance(feature_column, SparseColumn) or isinstance(feature_column, VarLenColumn):
                embedding_layer = keras.layers.Embedding(input_dim=feature_column.vocabulary_size,
                                                         output_dim=feature_column.embedding_dim)
                embedding_layer_dict[name] = embedding_layer

        return embedding_layer_dict

    def embedding_lookup(self, inputs):
        embedding_dict = {}
        for name, feature_column in self.feature_map.items():
            embedding = self.embedding_layer_dict[name](inputs[name])
            embedding_dict[name] = embedding

        return embedding_dict

    def concat_grouped_embedding(self, embedding_dict, group_name="general"):
        embedding_list = []
        for name, feature_columnn in self.feature_map.items():
            if feature_columnn.group_name == group_name or group_name=="general":
                embedding_list.append(embedding_dict[name])

        concated_embedding = tf.concat(embedding_list, axis=1)

        return concated_embedding

    def call(self, inputs):
        # 查embeddng
        embedding_dict = self.embedding_lookup(inputs)

        # general_input
        general_embedding = self.concat_grouped_embedding(embedding_dict, group_name="general")

        # ppnet input
        user_embedding = self.concat_grouped_embedding(embedding_dict, group_name="user")
        item_embedding = self.concat_grouped_embedding(embedding_dict, group_name="item")

        # epnet input
        domain_embedding = self.concat_grouped_embedding(embedding_dict, group_name="domain")

        task_logits = []
        for task_idx in range(self.task_num):
            logit = self.task_towers[task_idx](general_embedding)
            task_logits.append(logit)

        return task_logits

if __name__ == "__main__":
    feature_map = {"mid": SparseColumn("mid", 100, 10, "user"),
                   "iid": SparseColumn("iid", 100, 10, "item"),
                   "scene": SparseColumn("scene", 100, 10, "domain")}

    model = PEPNet(feature_map, 1, [100, 64], 1)

    inputs = {"mid": tf.constant([[1], [2]]), "iid": tf.constant([[1], [2]]), "scene": tf.constant([[1], [2]])}

    print(model(inputs))