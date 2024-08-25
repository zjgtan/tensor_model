import pandas as pd

# 创建一个读取器，每次读取 1000 行
chunksize = 1000
for chunk in pd.read_csv('/home/chenjiawei/data/train.txt', chunksize=chunksize, sep="\t"):
    # 对每个数据块进行处理
    print(chunk.head())
    break
