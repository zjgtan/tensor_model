
class SparseColumn:
    def __init__(self, name, group):
        """稀疏特征列
        name: 特征名
        vocabulary_size: 词表大小
        embedding_dim: embedding层大小
        group_name: 特征组
        """
        self.name = name
        self.group = group


class VarLenColumn:
    def __init__(self, name, vocabulary_size, embedding_dim, max_len, group_name):
        """稀疏特征列
        name: 特征名
        vocabulary_size: 词表大小
        max_len: 最大长度
        embedding_dim: embedding层大小
        group_name: 特征组
        """
        self.name = name
        self.max_len = max_len
        self.vocabulary_size = vocabulary_size
        self.group_name = group_name
        self.embedding_dim = embedding_dim



def get_feature_map_from_yaml_config(yaml_config):
    feature_map = {}

    for idx in range(len(yaml_config["feature_map"]["name"])):
        name = yaml_config["feature_map"]["name"][idx]
        type = yaml_config["feature_map"]["type"][idx]
        group = yaml_config["feature_map"]["group"][idx]

        if type == "sparse":
            feature_map[name] = SparseColumn(name, group)

    return feature_map