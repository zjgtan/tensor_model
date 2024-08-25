import tensorflow as tf
import pandas as pd

integer_fields = ["I{}".format(i) for i in range(1, 14)]
category_fields = ["C{}".format(i) for i in range(1, 27)]

columns = ["label"] + integer_fields + category_fields


df = pd.read_csv("/home/chenjiawei/data/sample.txt", sep="\t", names=columns)

# 缺失值处理
# 连续值
for feature in integer_fields:
    df[feature].fillna(df[feature].median(), inplace=True) # 中位数补全
# 分类值
for feature in category_fields:
    df[feature].fillna("missing", inplace=True) #固定值补全

def input_function():
    features = df.drop("label", axis=1)
    labels = df["label"]
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.batch(1)
    return dataset

def model_fn(features, labels, mode, params):
    #net = tf.feature_column.input_layer(features, params['feature_columns'])
    feature_column_dict = dict([(field, tf.feature_column.numeric_column(key=field, shape=(1))) \
        for field in integer_fields] + \
            [(field, tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(key=field, hash_bucket_size=1000), dimension=8)) \
                for field in category_fields])

    dnn_input_list = []
    for field in category_fields:
        emb = tf.feature_column.input_layer(features, [feature_column_dict[field]])
        dnn_input_list.append(emb)

    net = tf.concat(dnn_input_list, axis=-1)

    # 打印形状变量
    #net_shape = tf.shape(net)
    #net = tf.compat.v1.Print(net, [net_shape], "net shape: ")

    
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    predicted_classes = tf.argmax(logits, 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    #  metrics 指标计算。
    accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
    
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
        
    
    
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):


    # 3 Define an estimator
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 2
        })

    classifier.train(
        input_fn=lambda: input_function()
    )

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda:input_function()
    )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == "__main__":
    tf.set_random_seed(1103) # avoiding different result of random
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


'''遍历数据集
iter = dataset.make_initializable_iterator()
next_element = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    while True:
        a = sess.run(next_element)
        print(a)
'''
