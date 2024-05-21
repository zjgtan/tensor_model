import tensorflow as tf
from model.AutoInt import AutoInt


fix_len_feature_columns = ["gender", "age", "job", "zipcode", "year"]
var_len_feature_columns = ["genre"]
self_attention_linear_projection_num_units = 64
self_attention_num_heads = 4


def parse_example(record):
    schema = {
        "gender": tf.io.FixedLenFeature((1, ), tf.int64),
        "age": tf.io.FixedLenFeature((1, ), tf.int64),
        "job": tf.io.FixedLenFeature((1, ),tf.int64),
        "zipcode": tf.io.FixedLenFeature((1, ),tf.int64),
        "year": tf.io.FixedLenFeature((1, ),tf.int64), 
        "genre": tf.io.RaggedFeature(tf.int64),
        "label": tf.io.FixedLenFeature((1, ),tf.int64)
    }
    parsed_example = tf.io.parse_single_example(record, schema)
    parsed_example["genre"] = tf.ragged.stack([parsed_example['genre']], axis=0)

    return parsed_example


movielen_dataset = tf.data.TFRecordDataset(["./output/ml-1m/movielens.tfrecord"]).map(parse_example)
movielen_dataset = movielen_dataset.batch(3)

model = AutoInt(vocab_size=400, emb_size=3, self_attention_linear_projection_num_units=64, self_attention_num_heads=4)
optimizer = tf.keras.optimizers.Adam()




for idx, batch in enumerate(movielen_dataset):
    with tf.GradientTape() as tape:
        logits = model(batch, fix_len_feature_columns, var_len_feature_columns)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = batch["label"], logits=logits))
        print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    

