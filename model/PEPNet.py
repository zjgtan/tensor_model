import tensorflow as tf
from tensorflow import keras
from feature_column import *

class GateNU(keras.Model):
    def __init__(self, hidden_units):
        super().__init__()

        layers = [
            keras.layers.Dense(hidden_units[0], "relu"),
            keras.layers.Dense(hidden_units[1], "sigmoid")
        ]
        
        self.gate_layer = keras.Sequential(layers)
        
    def call(self, inputs):
        return self.gate_layer(inputs) * 2


class EPNet(keras.Model):
    def __init__(self, hidden_units, feature_columns):
        super().__init__()
        self.gate_unit = GateNU(hidden_units)
        self.feature_columns = feature_columns

    def call(self, domain_embedding, general_embedding):
        input_embedding = tf.concat([domain_embedding, tf.stop_gradient(general_embedding)], axis=-1)
        gate = self.gate_unit(input_embedding)
        output = tf.tile(gate, [1, len(self.feature_columns)]) * general_embedding
        return output


class PEPNet(keras.Model):
    def __init__(self, feature_map, embedding_dim, task_num, task_hidden_units, domain_num, epnet_hidden_units):
        super().__init__()
        self.feature_map = feature_map # 包含了模型用到的所有特征列信息
        self.embedding_dim = embedding_dim
        self.task_num = task_num
        self.embedding_layer_dict = self.create_embedding_layer(feature_map)

        self.epnet = EPNet(epnet_hidden_units + [embedding_dim])

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
            embedding_dict[name] = embedding

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

        # epnet input

        task_logits = []
        for task_idx in range(self.task_num):
            logit = self.task_towers[task_idx](general_embedding)
            task_logits.append(logit)

        return task_logits

if __name__ == "__main__":
    domain_embedding = tf.constant([[1.0, 2.0]])
    general_embedding = tf.constant([[3.0, 4.0]])
    epnet = EPNet([64, 2])

    gate0 = epnet(domain_embedding, general_embedding)

    print(tf.tile(gate0, [1, 5]))
    