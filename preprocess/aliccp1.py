import re
import os
import gc
import time
import joblib
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random

random.seed(2022)
np.random.seed(2022)
sample_skeleton_train_path = '../mtl/sample_skeleton_train.csv'
common_features_train_path = '../mtl/common_features_train.csv'
sample_skeleton_test_path = '../mtl/sample_skeleton_test.csv'
common_features_test_path = '../mtl/common_features_test.csv'
save_path = "../mtl/"
write_features_map_path = save_path + 'features_map.pkl'
write_features_path = save_path + 'all_features'
sparse_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210', '216', '508', '509', '702', '853', '301', '109_14', '110_14', '127_14', '150_14']
dense_columns = ['109_14', '110_14', '127_14', '150_14', '508', '509', '702', '853']
uses_columns = [col for col in sparse_columns] + ['D' + col for col in dense_columns]

padding_idx = 10091379


def preprocess_data(mode='train'):
    assert mode in ['train', 'test']
    if mode == "test" and not os.path.exists(write_features_map_path):
        print("Error! Please run the train mode first!")
        return
    common_features_path = common_features_train_path if mode == "train" else common_features_test_path
    sample_skeleton_path = sample_skeleton_train_path if mode == "train" else sample_skeleton_test_path

    print(f"Start processing common_features_{mode}")
    common_feat_dict = {}
    with open(common_features_path, 'r') as fr:
        for line in tqdm(fr):
            line_list = line.strip().split(',')
            feat_strs = line_list[2]
            feat_dict = {}
            for fstr in feat_strs.split('\x01'):
                field, feat_val = fstr.split('\x02')
                feat, val = feat_val.split('\x03')
                if field in sparse_columns:
                    feat_dict.setdefault(field, [])
                    feat_dict[field].append(int(feat))
            common_feat_dict[line_list[0]] = feat_dict

    print('join feats...')
    with tf.io.TFRecordWriter(f"../mtl/{mode}.tfrecord") as tfd_writer:
        with open(sample_skeleton_path, 'r') as fr:
            for line in tqdm(fr):
                line_list = line.strip().split(',')
                if line_list[1] == '0' and line_list[2] == '1':
                    continue
                if line_list[1] == '0' and random.random() > 0.1:
                    continue
                feat_strs = line_list[5]
                feat_dict = {}
                for fstr in feat_strs.split('\x01'):
                    field, feat_val = fstr.split('\x02')
                    feat, val = feat_val.split('\x03')
                    if field in sparse_columns:
                        feat_dict.setdefault(field, [])
                        feat_dict[field].append(int(feat))
                feat_dict.update(common_feat_dict[line_list[3]])
                feat_dict["click"] = [int(line_list[1])]
                feat_dict["conversion"] = [int(line_list[2])]

                tf_record = {}
                for idx, slot in enumerate(sparse_columns):
                    if slot in feat_dict:
                        tf_record[slot] = tf.train.Feature(int64_list=tf.train.Int64List(value=feat_dict[slot]))
                    else:
                        tf_record[slot] = tf.train.Feature(int64_list=tf.train.Int64List(value=[padding_idx + idx]))

                tf_record["click"] = tf.train.Feature(float_list=tf.train.FloatList(value=feat_dict["click"]))
                tf_record["conversion"] = tf.train.Feature(float_list=tf.train.FloatList(value=feat_dict["conversion"]))

                example = tf.train.Example(features=tf.train.Features(feature=tf_record))

                tfd_writer.write(example.SerializeToString())

if __name__ == "__main__":
    preprocess_data(mode='train')
    preprocess_data(mode='test')



