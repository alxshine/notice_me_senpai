import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 750, 1000, 1])

    #conv1 block
    conv11 = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm11 = tf.layers.batch_normalization(conv11)

    conv12 = tf.layers.conv2d(
            inputs=norm11,
            filters=64,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm12 = tf.layers.batch_normalization(conv12)

    pool1 = tf.layers.max_pooling2d(norm12,
            [2,2],
            [2,2],
            'same')

    #conv2 block
    conv21 = tf.layers.conv2d(
            inputs=pool1,
            filters=128,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm21 = tf.layers.batch_normalization(conv21)

    conv22 = tf.layers.conv2d(
            inputs=norm21,
            filters=128,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm22 = tf.layers.batch_normalization(conv22)

    pool2 = tf.layers.max_pooling2d(norm22,
            [2,2],
            [2,2],
            'same')

    #conv3 block
    conv31 = tf.layers.conv2d(
            inputs=pool2,
            filters=256,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm31 = tf.layers.batch_normalization(conv31)

    conv32 = tf.layers.conv2d(
            inputs=norm31,
            filters=256,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm32 = tf.layers.batch_normalization(conv32)
    
    conv33 = tf.layers.conv2d(
            inputs=norm32,
            filters=256,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm33 = tf.layers.batch_normalization(conv33)
    
    pool3 = tf.layers.max_pooling2d(norm33,
            [2,2],
            [2,2],
            'same')

    #conv4 block
    conv41 = tf.layers.conv2d(
            inputs=pool3,
            filters=512,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm41 = tf.layers.batch_normalization(conv41)

    conv42 = tf.layers.conv2d(
            inputs=norm41,
            filters=512,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm42 = tf.layers.batch_normalization(conv42)
    
    conv43 = tf.layers.conv2d(
            inputs=norm42,
            filters=512,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm43 = tf.layers.batch_normalization(conv43)
    
    pool4 = tf.layers.max_pooling2d(norm43,
            [2,2],
            [2,2],
            'same')

    #conv5 block
    conv51 = tf.layers.conv2d(
            inputs=pool4,
            filters=512,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm51 = tf.layers.batch_normalization(conv51)

    conv52 = tf.layers.conv2d(
            inputs=norm51,
            filters=512,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm52 = tf.layers.batch_normalization(conv52)
    
    conv53 = tf.layers.conv2d(
            inputs=norm52,
            filters=512,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu)
    norm53 = tf.layers.batch_normalization(conv53)
    
    pool5 = tf.layers.max_pooling2d(norm53,
            [2,2],
            [2,2],
            'same')

    #conv6-7 block
    conv61 = tf.layers.conv2d(
            inputs=pool5,
            filters=4096,
            kernel_size=[7,7],
            padding='same',
            activation=tf.nn.relu)
    drop61 = tf.layers.dropout(
            conv61)

    conv62 = tf.layers.conv2d(
            inputs=drop61,
            filters=4096,
            kernel_size=[1,1],
            padding='same',
            activation=tf.nn.relu)
    drop62 = tf.layers.dropout(
            conv62)

    conv63 = tf.layers.conv2d(
            inputs=drop62,
            filters=2,
            kernel_size=[1,1],
            padding='same')
    deconv63 = tf.layers.conv2d_transpose(
            inputs=conv63,
            filters=1,
            kernel_size=[1,1],
            padding='same')
    upsample63 = tf.image.resize_images(deconv63, [46, 62])

    #first merge
    convm1 = tf.layers.conv2d(
            inputs=pool4,
            filters=1,
            kernel_size=[2,2],
            padding='same')
    cropm1 = tf.image.resize_image_with_crop_or_pad(convm1, 46, 62)
    merged1 = tf.add(cropm1, upsample63)
    upsamplem1 = tf.image.resize_images(merged1, [92, 124])

    #second merge
    convm2 = tf.layers.conv2d(
            inputs=pool3,
            filters=1,
            kernel_size=[2,2],
            padding='same')
    cropm2 = tf.image.resize_image_with_crop_or_pad(convm2, 92, 124)
    merged2 = tf.add(cropm2, upsamplem1)
    upsamplem2 = tf.image.resize_images(merged2, [750, 1000])

    output = upsamplem2
    
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
    features = np.load("../dataset/images.npy")
    print("Done")
    print("Loading truth maps...")
    maps = np.load("../dataset/maps.npy")
    print("Done")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": features},
            y=maps,
            batch_size=1,
            num_epochs=10,
            shuffle=True)
    print("Training classifier...")
    classifier.train(
            input_fn=train_input_fn)
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
