import tensorflow as tf
import joblib
from sklearn.metrics import roc_auc_score
from model.feature_column import *
from model.PEPNet import PEPNet
import yaml


"""
alicpp数据格式
    sample_skeleton: sampleID (样本编码),click,conversion,common_feature_index,特征域数量, 特征列表
        特征列表：field_id, feature_id, feature_value
    common_feature_index: 主要是用户侧特征
        index,特征数,特征列表：field_id, feature_id, feature_value

"""

def parse_example(record, feature_map):
    schema = {}
    for feat, col in feature_map.items():
        if isinstance(col, SparseColumn):
            schema[feat] = tf.io.FixedLenFeature((1, ), tf.int64)
        elif isinstance(col, VarLenColumn):
            schema[feat] = tf.io.RaggedFeature(tf.int64)

    parsed_example = tf.io.parse_single_example(record, schema)
    for feat, col in feature_map.items():
        if isinstance(col, VarLenColumn):
            parsed_example[feat] = tf.ragged.stack([parsed_example[feat]], axis=0)

    return parsed_example

def train_epoch(model, dataset, optimizer):

    for idx, batch in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = 

    
    




if __name__ == "__main__":
    with open("./config/pepnet.yaml", "r") as fd:
        yaml_config = yaml.safe_load(fd)

    feature_map = get_feature_map_from_yaml_config(yaml_config)

    # 定义模型结构

    net = PEPNet(feature_map, yaml_config["vocab_size"], yaml_config["embedding_dim"], yaml_config["task_num"], yaml_config["dnn_hidden_units"], yaml_config["gate_hidden_dim"])


    # 定义输入输出数据流
    alicpp_train_set = tf.data.TFRecordDataset(["../mtl/train.tfrecord"]).map(lambda record: parse_example(record, feature_map)).batch(100)

    for idx, batch in enumerate(alicpp_train_set):
        print(batch)
        break




