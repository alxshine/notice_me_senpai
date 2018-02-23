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
    dense1 = tf.layers.dense(flat, 15*20*20)
    dense2 = tf.layers.dense(dense1, 15*20*20)
    dense3 = tf.layers.dense(dense2, 15*20*20)
    unflat = tf.reshape(dense3, [-1, 15, 20, 20])


    dc1 = tf.layers.conv2d_transpose(
            unflat,
            20,
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
    features = np.load("../dataset/extracted.npy")
    print("Done")
    print("Loading truth maps...")
    maps = np.load("../dataset/maps.npy")
    print("Done")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features[:780]},
            y=maps[:780],
            batch_size=3,
            num_epochs=10,
            shuffle=True)
    # print("Training classifier...")
    # classifier.train(
            # input_fn=train_input_fn,
            # steps=20000)
    # print("Done")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features[780:]},
            y=maps[780:],
            num_epochs=1,
            batch_size=3,
            shuffle=False)
    # print("Evaluating...")
    # eval_results = classifier.evaluate(input_fn=eval_input_fn)
    # print("Done, results: {}".format(eval_results))
    
    predictions = classifier.predict(eval_input_fn)
    incorrect_pixels = 0
    total_pixels = 0
    index = 780
    for p in predictions:
        pred = p['classes']
        pred[pred>pred.mean()] = 1
        pred[pred<1] = 0

        incorrect_pixels += np.count_nonzero(pred-maps[index:index+1])
        total_pixels += np.prod(pred.shape)

        if incorrect_pixels/total_pixels < 0.2:
            plt.figure()
            plt.subplot(131)
            plt.imshow(maps[index,:,:,0],cmap='gray')
            plt.title("truth")

            plt.subplot(132)
            plt.imshow(pred[:,:,0],cmap='gray')
            plt.title("prediction")

            diff = pred - maps[index]
            plt.subplot(133)
            plt.imshow(diff[:,:,0])
            plt.colorbar()
            plt.title("diff")
            
            incorrect_rate = np.sum(np.abs(diff))/np.prod(diff.shape)
            print("Accuracy: {}%".format((1-incorrect_rate)*100))

            plt.show()
        index += 1

    incorrect_rate = incorrect_pixels/total_pixels
    print("Accuracy: {}%".format((1-incorrect_rate)*100))

if __name__ == "__main__":
    main(sys.argv)
