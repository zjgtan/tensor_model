import sys
sys.path.append(".")
import tensorflow as tf
from tensorflow import keras
from model.feature_column import *



class CGC(keras.Model):
    def __init__(self, num_specific_experts, num_shared_experts, expert_hidden_units, gate_hidden_units, task_num):
        super().__init__()
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.expert_hidden_units = expert_hidden_units
        self.gate_hidden_units = gate_hidden_units
        self.task_num = task_num

        self.specific_experts = []
        for _ in range(task_num):
            for _ in range(self.num_specific_experts):
                self.specific_experts.append(self.get_mlp_block(expert_hidden_units, ["relu"] * len(expert_hidden_units)))

        self.shared_experts = []
        for _ in range(self.num_shared_experts):
            self.shared_experts.append(self.get_mlp_block(expert_hidden_units, ["relu"] * len(expert_hidden_units)))

        self.specific_gates = []
        for _ in range(self.task_num + 1):
            self.specific_gates.append(self.get_mlp_block(gate_hidden_units + [self.num_specific_experts + self.num_shared_experts], ["relu"] * len(gate_hidden_units) + ["softmax"]))

        self.shared_gates = []
        self.shared_gates.append(self.get_mlp_block(gate_hidden_units + [self.num_specific_experts * self.task_num + self.num_shared_experts], ["relu"] * len(gate_hidden_units) + ["softmax"]))
        

    def get_mlp_block(self, hidden_units, activations):
        mlp = keras.Sequential()
        for idx in range(len(hidden_units)):
            mlp.add(keras.layers.Dense(hidden_units[idx], activation=activations[idx]))

        return mlp    

    def call(self, inputs):

        specific_expert_outputs = []
        for task_idx in range(self.task_num):
            for expert_idx in range(self.num_specific_experts):
                specific_expert_outputs.append(self.specific_experts[task_idx * self.num_specific_experts + expert_idx](inputs[task_idx]))


        shared_expert_outputs = []
        for expert_idx in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[expert_idx](inputs[-1]))

        specific_gate_outputs = []
        for task_idx in range(self.task_num):
            specific_gate_outputs.append(self.specific_gates[task_idx](inputs[task_idx]))


        shared_gate_outputs = []
        shared_gate_outputs.append(self.shared_gates[0](inputs[-1]))

        cgc_outputs = []
        for task_idx in range(self.task_num):
            merged_experts = tf.concat([e[:, tf.newaxis, :] for e in specific_expert_outputs[task_idx * self.num_specific_experts : (task_idx+1) * self.num_specific_experts] + shared_expert_outputs], axis=1)
            merged_experts = tf.squeeze(tf.matmul(merged_experts, tf.expand_dims(specific_gate_outputs[task_idx], axis=-1), transpose_a=True), axis=-1)

            cgc_outputs.append(merged_experts)

        # shared gate
        merged_experts = tf.concat([e[:, tf.newaxis, :] for e in specific_expert_outputs + shared_expert_outputs], axis=1)
        merged_experts = tf.squeeze(tf.matmul(merged_experts, tf.expand_dims(shared_gate_outputs[0], axis=-1), transpose_a=True), axis=-1)
        cgc_outputs.append(merged_experts)

        return cgc_outputs


class PLE(keras.Model):
    def __init__(self, feature_map, vocab_size, embedding_dim, task_num, num_specific_experts, num_shared_experts, expert_hidden_units, gate_hidden_units, num_level, tower_hidden_units):
        super().__init__()
        self.feature_map = feature_map
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.task_num = task_num
        
        self.num_level = num_level
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.expert_hidden_units = expert_hidden_units
        self.gate_hidden_units = gate_hidden_units
        self.tower_hidden_units = tower_hidden_units

        self.embedding_layer = self.create_embedding_layer(vocab_size)

        self.cgc_layers = []
        for _ in range(num_level):
            self.cgc_layers.append(CGC(self.num_specific_experts, self.num_shared_experts, self.expert_hidden_units, self.gate_hidden_units, self.task_num))

        self.towers = [self.get_mlp_block(tower_hidden_units, ["relu"] * (len(tower_hidden_units) - 1) + ["linear"]) for _ in range(task_num)]


    def create_embedding_layer(self, vocab_size):
        embedding_layer = keras.layers.Embedding(input_dim=vocab_size,
                                                    output_dim=self.embedding_dim)
        return embedding_layer

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
    
    def get_mlp_block(self, hidden_units, activations):
        mlp = keras.Sequential()
        for idx in range(len(hidden_units)):
            mlp.add(keras.layers.Dense(hidden_units[idx], activation=activations[idx]))

        return mlp   

    def call(self, inputs):
        embedding_dict = self.embedding_lookup(inputs)
        concated_embedding = self.concat_embedding(embedding_dict)

        cgc_inputs = [concated_embedding] * (self.task_num + 1)
        cpc_outputs = cgc_inputs
        for level in range(self.num_level):
            cgc_outputs = self.cgc_layers[level](cgc_inputs)
            cgc_inputs = cgc_outputs


        multi_task_logits = []
        for idx in range(self.task_num):
            tower_logits = self.towers[idx](cpc_outputs[idx])
            multi_task_logits.append(tower_logits)

        return multi_task_logits


if __name__ == "__main__":
    feature_map = {"mid": SparseColumn("mid", "user"),
            "iid": SparseColumn("iid", "item"),
            "scene": SparseColumn("scene", "domain")}

    model = PLE(feature_map=feature_map,
                vocab_size=10,
                embedding_dim=4,
                task_num=2,
                num_specific_experts=2,
                num_shared_experts=2,
                expert_hidden_units=[32, 32],
                gate_hidden_units=[32],
                tower_hidden_units=[32, 1],
                num_level=2)
        
    batch = {"mid": tf.constant([[1], [2]]),
             "iid": tf.constant([[1], [2]]),
             "scene": tf.constant([[1], [2]])}

    print(model(batch))