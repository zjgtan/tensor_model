import tensorflow as tf
import joblib
from sklearn.metrics import roc_auc_score
import yaml


"""
alicpp数据格式
    sample_skeleton: sampleID (样本编码),click,conversion,common_feature_index,特征域数量, 特征列表
        特征列表：field_id, feature_id, feature_value
    common_feature_index: 主要是用户侧特征
        index,特征数,特征列表：field_id, feature_id, feature_value

"""



if __name__ == "__main__":
    with open("../config/pepnet.yaml", "r") as fd:
        model_config = yaml.safe_load(fd)

    print(model_config)
