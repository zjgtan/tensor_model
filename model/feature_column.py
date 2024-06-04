
class SparseColumn:
    def __init__(self, name, vocabulary_size, embedding_dim, group_name):
        """稀疏特征列
        name: 特征名
        vocabulary_size: 词表大小
        embedding_dim: embedding层大小
        group_name: 特征组
        """
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.group_name = group_name


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

