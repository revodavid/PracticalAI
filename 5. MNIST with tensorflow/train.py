import os
import sys
import argparse
import tensorflow as tf
from misc.helpers import *
from misc.digits import Digits

###################################################################
# Model                                                           #
###################################################################
@print_info
def linear_model(x):
    with tf.name_scope("Model"):
        pred = tf.layers.dense(inputs=x, units=10, 
                               activation=tf.nn.softmax)
        return tf.identity(pred, name="prediction")

@print_info
def cnn_model(x):
    conv1 = tf.layers.conv2d(inputs=tf.reshape(x, [-1, 28, 28, 1]), 
                             filters=32, 
                             kernel_size=[5, 5], 
                             padding="same", 
                             activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    with tf.name_scope('Model'):
        pred = tf.layers.dense(inputs=dense, units=10, activation=tf.nn.softmax)
        return tf.identity(pred, name="prediction")

###################################################################
# Training                                                        #
###################################################################
@print_info
def train_model(x, y, cost, optimizer, accuracy, learning_rate, batch_size, epochs, data_dir, outputs_dir, logs_dir):

    # get run
    try:
        run = Run.get_context()
    except:
        run = None

    # log paramters
    aml_log(run, learning_rate=learning_rate,
            batch_size=batch_size, epochs=epochs,
            data_dir=data_dir, outputs_dir=outputs_dir,
            logs_dir=logs_dir)

    info('Initializing Devices')
    print(' ')
    
    # load MNIST data (if not available)
    digits = Digits(data_dir, batch_size)
    test_x, test_y = digits.test
    
    # Create a summary to monitor cost tensor
    tf.summary.scalar("cost", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(str(logs_dir), graph=tf.get_default_graph())

    # Initializing the variables
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        acc = 0.
        info('Training')

        # epochs to run
        for epoch in range(epochs):
            print("Epoch {}".format(epoch+1))
            avg_cost = 0.
            # loop over all batches
            for i, (train_x, train_y) in enumerate(digits):

                # Run optimization, cost, and summary
                _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                         feed_dict={x: train_x, y: train_y})

                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * digits.total + i)
                # Compute average loss
                avg_cost += c / digits.total
                print("\r    Batch {}/{} - Cost {:5.4f}".format(i+1, digits.total, avg_cost), end="")

            acc = accuracy.eval({x: test_x, y: test_y})
            print("\r    Cost: {:5.4f}, Accuracy: {:5.4f}\n".format(avg_cost, acc))
            # aml log
            aml_log(run, cost=avg_cost, accuracy=acc)
        
        # save model
        info("Saving Model")
        save_model(sess, outputs_dir, 'Model/prediction')

def main(settings):
    # resetting graph
    tf.reset_default_graph()

    # mnist data image of shape 28*28=784
    x = tf.placeholder(tf.float32, [None, 784], name='x')

    # 0-9 digits recognition => 10 classes
    y = tf.placeholder(tf.float32, [None, 10], name='y')

    # model
    hx = cnn_model(x)

    # accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hx, 1), tf.argmax(y, 1)), tf.float32))

    # cost / loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=hx))

    # optimizer
    optimizer = tf.train.AdamOptimizer(settings.lr).minimize(cost)

    # training session
    train_model(x, y, cost, optimizer, accuracy, 
        settings.lr, settings.batch, settings.epochs, 
        settings.data, settings.outputs, settings.logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=100, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)
    parser.add_argument('-g', '--logs', help='log directory', default='logs')
    parser.add_argument('-o', '--outputs', help='output directory', default='outputs')
    args = parser.parse_args()

    args.data = check_dir(args.data).resolve()
    args.outputs = check_dir(args.outputs).resolve()
    args.logs = check_dir(args.logs).resolve()
    
    main(args)