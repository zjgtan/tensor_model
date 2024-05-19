
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

"""
movies.dat
    1::Toy Story (1995)::Animation|Children's|Comedy
ratings.dat
    1::1193::5::978300760
users.dat:
    1::F::1::10::48067
输出样本：
    uid iid label male age occupation zipcode year genre
"""


year_dict = {}
for i in range(1919, 1930):
	year_dict[i] = 1
for i in range(1930, 1940):
	year_dict[i] = 2
for i in range(1940, 1950):
	year_dict[i] = 3
for i in range(1950, 1960):
	year_dict[i] = 4
for i in range(1960, 1970):
	year_dict[i] = 5
for i in range(1970, 2001):
	year_dict[i] = 6 + i - 1970

movie_dict = {}
with open("./dataset/ml-1m/movies.dat", encoding="ISO-8859-1") as fd:
    for line in fd:
        fields = line.rstrip().split("::")
        movie_id = fields[0]
        title = fields[1]
        year = int(title[-5:-1])
        year_bucket = year_dict[year]
        genres = fields[2].split("|")

        #movie_dict[movie_id] = ["year-{}".format(year_bucket), "iid-{}".format(movie_id)] + ["genre-{}".format(genre) for genre in genres]
        movie_dict[movie_id] = ["year#{}".format(year_bucket)] + ["genre#{}".format(genre) for genre in genres]


user_dict = {}
with open("./dataset/ml-1m/users.dat") as fd:
    for line in fd:
        fields = line.rstrip().split("::")
        user_id = fields[0]
        gender = fields[1]
        age = fields[2]
        job = fields[3]
        zipcode = fields[4]

        #user_dict[user_id] = ["uid-{}".format(user_id), "age-{}".format(age), "job-{}".format(job), "zipcode-{}".format(zipcode)]
        user_dict[user_id] = ["gender#{}".format(gender), "age#{}".format(age), "job#{}".format(job), "zipcode#{}".format(zipcode[:3])]


feature_dict = {}
for iid, features in movie_dict.items():
    for feature in features:
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

for uid, features in user_dict.items():
    for feature in features:
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

print(feature_dict)


with tf.io.TFRecordWriter("./output/ml-1m/movielens.tfrecord") as tfd_writer:
    with open("./dataset/ml-1m/ratings.dat") as fd:
        for line in fd:
            uid, iid, rate, ts = line.rstrip().split("::")
            if int(rate) == 3:
                continue
            label = 1 if int(rate) > 3 else 0

            user_feature = user_dict[uid]
            item_feautre = movie_dict[iid]

            record = {}
            for feature in user_feature + item_feautre:
                slot_name = feature.split("#")[0]
                slot_idx = feature_dict[feature]

                record.setdefault(slot_name, [])
                record[slot_name].append(slot_idx)

            record["label"] = [label]

            for slot_name in ["gender", "age", "job", "zipcode", "year", "genre", "label"]:
                record[slot_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=record[slot_name]))

            example = tf.train.Example(features=tf.train.Features(feature=record))

            tfd_writer.write(example.SerializeToString())

        
            
        


        





