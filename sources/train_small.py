import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 750, 1000, 1])

    conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=3,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(
            conv,
            [2,2],
            [2,2])

    conv2 = tf.layers.conv2d(
            pool1,
            7,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
            conv2,
            [5,5],
            [5,5])

    conv3 = tf.layers.conv2d(
            pool2,
            20,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(
            conv3,
            [5,5],
            [5,5])

    flat = tf.reshape(pool3, [-1, 15*20*20])
    dense = tf.layers.dense(flat, 15*20*20)
    unflat = tf.reshape(dense, [-1, 15, 20, 20])


    dc1 = tf.layers.conv2d_transpose(
            unflat,
            20,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    ups1 = tf.image.resize_images(dc1, [75, 100])

    dc2 = tf.layers.conv2d_transpose(
            ups1,
            7,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    # ups2 = tf.image.resize_images(dc2, [375, 500])

    # dc3 = tf.layers.conv2d_transpose(
            # ups2,
            # 3,
            # [3,3],
            # padding='same',
            # activation=tf.nn.relu)

    ups3 = tf.image.resize_images(dc2, [750, 1000])

    dc4 = tf.layers.conv2d_transpose(
            ups3,
            1,
            [3,3],
            padding='same',
            activation=tf.nn.relu)


    norm = tf.div(
            tf.subtract(dc4, tf.reduce_min(dc4)),
            tf.subtract(tf.reduce_max(dc4), tf.reduce_min(dc4)))

    output = norm
    
    predictions = {
            "classes": output,
            "probabilities": output
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.hinge_loss(labels, output)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    #create estimator
    print("Creating estimator...")
    classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn)
    print("Done")

    #load training data
    print("Loading features...")
    features = np.load("../dataset/extracted.npy")[6:7]
    print("Done")
    print("Loading truth maps...")
    maps = np.load("../dataset/maps.npy")[6:7]
    print("Done")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features},
            y=maps,
            batch_size=2,
            num_epochs=200,
            shuffle=True)
    print("Training classifier...")
    classifier.train(
            input_fn=train_input_fn,
            steps=200000)
    print("Done")

    print("Evaluating...")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features},
            y=maps,
            num_epochs=1,
            shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print("Done, results: {}".format(eval_results))
    
    predictions = classifier.predict(eval_input_fn)
    for p in predictions:
        plt.figure()
        plt.subplot(131)
        plt.imshow(features[0,:,:,0],cmap='gray')
        plt.title("features")

        plt.subplot(132)
        plt.imshow(maps[0,:,:,0],cmap='gray')
        plt.title("map")

        plt.subplot(133)
        plt.imshow(p['classes'][:,:,0],cmap='gray')
        plt.title("prediction")

        plt.show()

if __name__ == "__main__":
    main(sys.argv)
