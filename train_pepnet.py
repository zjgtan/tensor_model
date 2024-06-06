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



if __name__ == "__main__":
    with open("./config/pepnet.yaml", "r") as fd:
        yaml_config = yaml.safe_load(fd)

    feature_map = get_feature_map_from_yaml_config(yaml_config)

    # 定义模型结构

    net = PEPNet(feature_map, yaml_config["embedding_dim"], yaml_config["task_num"], yaml_config["dnn_hidden_units"], yaml_config["gate_hidden_dim"])


    # 定义输入输出数据流


