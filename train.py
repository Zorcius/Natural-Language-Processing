# -*- coding: utf-8 -*-
#coding=gbk
#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers
from text_cnn import TextCNN
import tqdm
import matplotlib.pyplot as plt
import random

# Parameters
# =======================================================
# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/pos.txt", "Data source for the positive data.")
#tf.flags.DEFINE_string("positive_data_file", "./data/pos.txt", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/", "Data source for the negative data.")
tf.flags.DEFINE_string("negative_data_file", "./data/neg.txt", "Data source for the negative data.")
tf.flags.DEFINE_integer("num_labels", 2, "Number of labels for data. (default: 2)")
tf.flags.DEFINE_string("stop_word_file","./data/stopWord.txt","Stop words for segementation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")#37 om
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 32, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")##可以修改
tf.flags.DEFINE_float("learning_rate", 1e-4, "network's learning rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 150, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Parse parameters from commands
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Prepare output directory for models and summaries
# =======================================================

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Data preprocess
# =======================================================
# Load data
print("Loading data...")
x_text, y = data_helpers.load_positive_negative_data_files(FLAGS.positive_data_file, FLAGS.negative_data_file,
                                                           cut=False,stop_words_list_file=None,) #不进行切分词
#x_text, y = data_helpers.load_positive_negative_data_files(FLAGS.positive_data_file, FLAGS.negative_data_file,
#                                                           cut=True, stop_words_list_file=FLAGS.stop_word_file) #切分词版本
#print(x_text)
# Get embedding vector
sentences = data_helpers.padding_sentences(x_text, '<PADDING>',padding_sentence_length=20)
x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = FLAGS.embedding_dim,file_to_save = os.path.join(out_dir, 'trained_word2vec.model')))
print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))

# Save params
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels' : FLAGS.num_labels, 'max_document_length' : 20}
data_helpers.saveDict(params, training_params_file)

# Shuffle data randomly
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

print("x_train.shape={}".format(x_train.shape))
print("y_train.shape={}".format(y_train.shape))

print("x_dev.shape={}".format(x_dev.shape))
print("y_dev.shape={}".format(y_dev.shape))

shuffle_index = np.arange(y_dev.shape[0])
#shuffle_index = np.random.permutation(np.arange(y_dev.shape[0]))

# Training
# =======================================================
#start_time = time.asctime( time.localtime(time.time()) )
start_time = datetime.datetime.now()
print('开始时间:'),
print(start_time)


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
	log_device_placement = FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        cnn = TextCNN(
	    sequence_length = x_train.shape[1],
	    num_classes = y_train.shape[1],
	    embedding_size = FLAGS.embedding_dim,
	    filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
	    #filter_sizes = FLAGS.filter_sizes,
	    num_filters = FLAGS.num_filters,
	    l2_reg_lambda = FLAGS.l2_reg_lambda)

	    # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            cnn.training_accuracies.append(accuracy)
            cnn.training_losses.append(loss)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        def plot_training_accuracies(*args, **kwargs):
            """
            对验证集数据的精度以图形化展示，方便知道模型收敛情况或者过拟合情况

            :param args: CNN对象，用于获取其中的training_accuracies参数
            :param kwargs:这里只需要用到batch_size这个参数
            """
            fig, ax = plt.subplots()
            batch_size = kwargs['batch_size']
            for cnn in args:
                ax.plot(range(0,len(cnn.training_accuracies)*batch_size,batch_size),
                cnn.training_accuracies, label=cnn.name) #x坐标轴进行标刻度，y坐标轴为每一次的accuracy值，每一个batch_size画一次图
            ax.set_xlabel('Training steps')
            ax.set_ylabel('Accuracy')
            ax.set_title('Validation Accuracy During Training')
            ax.legend(loc=4)
            ax.set_ylim([0,1])
            plt.yticks(np.arange(0, 1.1, 0.1))
            #plt.xticks(np.arange(0, len(cnn.training_accuracies)))
            plt.grid(True)
            plt.show()
            #plt.savefig("Validation Accuracy During Training")

        def plot_training_losses(*args, **kwargs):
            """
            对验证集数据的loss以图形化展示，方便知道模型收敛情况或者过拟合情况

            :param args: CNN对象，用于获取其中的training_accuracies参数
            :param kwargs:这里只需要用到batch_size这个参数
            """
            fig, ax = plt.subplots()
            batch_size = kwargs['batch_size']
            for cnn in args:
                ax.plot(range(0,len(cnn.training_losses)*batch_size,batch_size),
                        cnn.training_losses, label=cnn.name) #x坐标轴进行标刻度，y坐标轴为每一次的loss值，每一个batch_size画一次图
            ax.set_xlabel('Training steps')
            ax.set_ylabel('Losses')
            ax.set_title('Validation Losses During Training')
            ax.legend(loc=4)
            ax.set_ylim([0,1])
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.grid(True)
            plt.show()
            #plt.savefig("Validation Losses During Training")
        # Generate batches
        #batches = data_helpers.batch_iter(
        #    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        #print(len(list(batches)))length=63400

        # Training loop. For each batch...
        data = np.array(list(zip(x_train, y_train)))
        data_size = len(data)
        num_batches_per_epoch = int((data_size - 1) / FLAGS.batch_size) + 1 #根据batch_size来分割整个data_size

        print("\n")

        for epoch in tqdm.tqdm(range(FLAGS.num_epochs)):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]

            #sample_600_index = random.sample(set(shuffle_index),1000)
            sample_600_index = random.sample(set(shuffle_index),3000)

            for batch_num in range(num_batches_per_epoch):
                start_idx = batch_num * FLAGS.batch_size
                end_idx = min((batch_num + 1) * FLAGS.batch_size, data_size)

                x_batch, y_batch = zip(*shuffled_data[start_idx: end_idx])
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                #从loss图的分析结果来看，learning_rate=1e-4下，运行到5000步时，loos就开始出现震荡
                #所以此时降低learning_rate以更慢的速度继续学习。
                if current_step == 2000:
                    FLAGS.learning_rate = FLAGS.learning_rate / 100
                if current_step == 4000:
                    FLAGS.learning_rate = FLAGS.learning_rate / 1000
                if current_step == 7000:
                    FLAGS.learning_rate = FLAGS.learning_rate / 10000
                if current_step == 7500:
                    FLAGS.learning_rate = FLAGS.learning_rate / 100000

                if current_step % FLAGS.evaluate_every == 0:

                    print("\nEvalution:")#每50个step做一次验证
                    print("\nLearning rate:")
                    print(FLAGS.learning_rate)

                    x_batch_dev = x_dev[sample_600_index]
                    y_batch_dev = y_dev[sample_600_index]
                    dev_step(x_batch_dev, y_batch_dev, writer=dev_summary_writer)
                    #dev_step(x_dev, y_dev, writer=dev_summary_writer)
                if current_step % FLAGS.checkpoint_every == 0: #每100次保存一下模型
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

        stop_time = datetime.datetime.now()
        print('结束时间:')
        print(stop_time)
        #print("模型训练时间：{0:%.2f}小时".format((stop_time - start_time).seconds / 3600))
        print("Total hours:")
        print((stop_time - start_time).seconds / 3600)

        #绘出最后的训练精度图
        plot_training_accuracies(cnn, batch_size=FLAGS.evaluate_every)
        plot_training_losses(cnn, batch_size=FLAGS.evaluate_every)
        '''
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)#提取出当前global_step的值作为当前step
            if current_step % FLAGS.evaluate_every == 0:#每50个step做一次验证
                print("\nEvaluation:")

                #dev_step(x_dev, y_dev, writer=dev_summary_writer)
                dev_step(x_batch_dev, y_batch_dev, writer=dev_summary_writer)

            if current_step % FLAGS.checkpoint_every == 0: #每100次保存一下模型
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))'''

sess.close()
#stop_time = time.asctime( time.localtime(time.time()) )

