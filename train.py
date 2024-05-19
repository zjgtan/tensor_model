import tensorflow as tf

def parse_example(record):
    schema = {
        "gender": tf.io.FixedLenFeature([],tf.int64),
        "age": tf.io.FixedLenFeature([],tf.int64),
        "job": tf.io.FixedLenFeature([],tf.int64),
        "zipcode": tf.io.FixedLenFeature([],tf.int64),
        "year": tf.io.FixedLenFeature([],tf.int64), 
        "genre": tf.io.RaggedFeature(tf.int64),
        "label": tf.io.FixedLenFeature([],tf.int64)
    }
    parsed_example = tf.io.parse_single_example(record, schema)
    parsed_example["genre"] = tf.ragged.stack([parsed_example['genre']])

    return parsed_example


movielen_dataset = tf.data.TFRecordDataset(["./output/ml-1m/movielens.tfrecord"]).map(parse_example)
movielen_dataset = movielen_dataset.batch(10)


embedding_layer = tf.keras.layers.Embedding(1000, 3)


for batch in movielen_dataset:
    print(batch)
    emb = embedding_layer(batch["genre"])
    emb = tf.reduce_mean(emb, axis=-2).to_tensor()
    print(emb)
    break
