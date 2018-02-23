import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 750, 1000, 1])

    c1 = tf.layers.conv2d(
            input_layer,
            64444,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    c11 = tf.layers.conv2d(
            c1,
            64,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    p1 = tf.layers.max_pooling2d(
            c11,
            [2,2],
            [2,2],
            padding='same')

    c2 = tf.layers.conv2d(
            p1,
            128,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    c21 = tf.layers.conv2d(
            c2,
            128,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    p2 = tf.layers.max_pooling2d(
            c21,
            [5,5],
            [5,5],
            padding='same')

    c3 = tf.layers.conv2d(
            p2,
            256,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    c31 = tf.layers.conv2d(
            c3,
            64,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    p3 = tf.layers.max_pooling2d(
            c31,
            [5,5],
            [5,5],
            padding='same')

    flat = tf.reshape(p3, [-1, 64*15*20])
    d1 = tf.layers.dense(flat, 64*15*20)
    d2 = tf.layers.dense(d1, 64*15*20)
    d3 = tf.layers.dense(d2, 64*15*20)
    d4 = tf.layers.dense(d3, 64*15*20)
    unflat = tf.reshape(d4, [-1, 64, 15, 20])

    dc1 = tf.layers.conv2d_transpose(
            unflat,
            1,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    ups3 = tf.image.resize_images(dc1, [750, 1000])

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
    features = np.load("../dataset/patterns.npy")
    print("Done")
    print("Loading truth maps...")
    maps = np.load("../dataset/maps.npy")
    print("Done")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features[:780]},
            y=maps[:780],
            batch_size=3,
            num_epochs=1,
            shuffle=True)
    print("Training classifier...")
    classifier.train(
            input_fn=train_input_fn,
            steps=20000)
    print("Done")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features[:1]},
            y=maps[:1],
            num_epochs=1,
            batch_size=3,
            shuffle=False)
    # print("Evaluating...")
    # eval_results = classifier.evaluate(input_fn=eval_input_fn)
    # print("Done, results: {}".format(eval_results))
    
    predictions = classifier.predict(eval_input_fn)
    incorrect_pixels = 0
    total_pixels = 0
    index = 0
    for p in predictions:
        pred = p['classes']
        pred[pred>pred.mean()] = 1
        pred[pred<1] = 0

        incorrect_pixels += np.count_nonzero(pred-maps[index:index+1])
        total_pixels += np.prod(pred.shape)


        plt.figure()
        plt.subplot(131)
        plt.imshow(maps[index,:,:,0],cmap='gray')
        plt.title("truth")

        plt.subplot(132)
        plt.imshow(pred[:,:,0],cmap='gray')
        plt.title("prediction")

        plt.subplot(133)
        plt.imshow(features[index,:,:,0])
        plt.colorbar()
        plt.title("features")
        
        # incorrect_rate = np.sum(np.abs(diff))/np.prod(diff.shape)
        # print("Accuracy: {}%".format((1-incorrect_rate)*100))

        plt.show()
        index += 1

    incorrect_rate = incorrect_pixels/total_pixels
    print("Accuracy: {}%".format((1-incorrect_rate)*100))

if __name__ == "__main__":
    main(sys.argv)
