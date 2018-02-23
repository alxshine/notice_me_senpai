import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 750, 1000, 1])

    c1 = tf.layers.conv2d(
            input_layer,
            8,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    p1 = tf.layers.max_pooling2d(
            c1,
            [2,2],
            [2,2])

    c2 = tf.layers.conv2d(
            p1,
            16,
            [5,5],
            padding='same',
            activation=tf.nn.relu)

    p2 = tf.layers.max_pooling2d(
            c2,
            [5,5],
            [5,5])

    c3 = tf.layers.conv2d(
            p2,
            32,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    p3 = tf.layers.max_pooling2d(
            c3,
            [5,5],
            [5,5])

    c4 = tf.layers.conv2d(
            p3,
            3,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    flat = tf.reshape(c4, [-1, 15*20*3])
    d1 = tf.layers.dense(flat, 15*20*3)
    d2 = tf.layers.dense(d1, 15*20*3)
    unflat = tf.reshape(d2, [-1, 3, 15, 20])


    d1 = tf.layers.conv2d_transpose(
            unflat,
            1,
            [3,3],
            padding='same',
            activation=tf.nn.relu)

    u1 = tf.image.resize_images(
            d1,
            [750, 1000])

    last = u1
    # norm = tf.div(
            # tf.subtract(last, tf.reduce_min(last)),
            # tf.subtract(tf.reduce_max(last), tf.reduce_min(last)))

    output = last

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
    features = np.load("../dataset/correlations.npy")
    temp = np.zeros([10,750,1000,1])
    temp[:,:,:,0] = features[:,::2,::2]
    features = temp.astype('float32')
    print(features.shape)
    print("Done")
    print("Loading truth maps...")
    maps = np.load("../dataset/maps.npy").astype('float32')[:10]
    print(maps.shape)
    print("Done")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features[1:]},
            y=maps[1:],
            batch_size=3,
            num_epochs=10,
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
        print(pred)

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
