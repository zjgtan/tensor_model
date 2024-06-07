import tensorflow as tf
import joblib
from sklearn.metrics import roc_auc_score
from model.feature_column import *
from model.MMOE import MMOE
import yaml
from tqdm import tqdm


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

    schema["click"] = tf.io.FixedLenFeature((1,), tf.float32)
    schema["conversion"] = tf.io.FixedLenFeature((1,), tf.float32)


    parsed_example = tf.io.parse_single_example(record, schema)
    for feat, col in feature_map.items():
        if isinstance(col, VarLenColumn):
            parsed_example[feat] = tf.ragged.stack([tf.ragged.stack([parsed_example[feat]], axis=0)], axis=0)

    return parsed_example

def train_epoch(model, dataset, optimizer):

    idx = 0

    for batch in tqdm(dataset):
        with tf.GradientTape() as tape:
            click_logits, conversion_logits = model(batch)
            click_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch["click"], logits=click_logits))
            conversion_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch["conversion"], logits=conversion_logits))


            click_auc = roc_auc_score(batch["click"], tf.nn.sigmoid(click_logits))

            loss = click_loss + conversion_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if idx % 100 == 0:
            print(loss, click_auc)


if __name__ == "__main__":
    with open("./config/mmoe.yaml", "r") as fd:
        yaml_config = yaml.safe_load(fd)

    feature_map = get_feature_map_from_yaml_config(yaml_config)

    # 定义模型结构
    net = MMOE(feature_map, yaml_config["vocab_size"], yaml_config["embedding_dim"], yaml_config["task_num"], yaml_config["num_experts"], yaml_config["expert_hidden_units"], yaml_config["gate_hidden_units"], yaml_config["tower_hidden_units"])

    optimizer = tf.keras.optimizers.Adam()

    # 定义输入输出数据流
    alicpp_train_set = tf.data.TFRecordDataset(["../mtl/train.tfrecord"]).map(lambda record: parse_example(record, feature_map)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size=yaml_config["batch_size"]))

    for epoch in range(yaml_config["epoch"]):
        train_epoch(net, alicpp_train_set, optimizer)