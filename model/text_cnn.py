# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class TextCNN(object):
    '''
    A CNN for text classification
    Uses and embedding layer, followed by a convolutional, max-pooling and softmax layer.
    '''

    def __init__(
            self, sequence_length, num_classes,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.name = "Text CNN"
        self.input_x = tf.placeholder(
            tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        self.training_accuracies = []
        self.training_losses = []
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:1'), tf.name_scope("embedding"):
            self.embedded_chars = self.input_x
            self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution layer 1
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expended,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    #padding="VALID",
                    name="conv")
                #apply batch normalization
                bn = tf.layers.batch_normalization(conv)
                # Apply nonlinearity
                conv1_out = tf.nn.relu(tf.nn.bias_add(bn, b), name="relu")

                ##Convolution layer 2
                hidden_W = tf.Variable(tf.truncated_normal([20, 1, num_filters, 16],stddev=0.1),name="hidden_W")#按context word
                hidden_b = tf.Variable(tf.constant(0.1, shape=[16]),name="hiden_b")
                conv_hidden_2 = tf.nn.conv2d(
                    conv1_out, hidden_W, strides=[1,1,1,1],padding="VALID",name="hidden_conv"
                )
                bn_hidden_2 = tf.layers.batch_normalization(conv_hidden_2)

                conv_hidden_2_out = tf.nn.relu(tf.nn.bias_add(bn_hidden_2,hidden_b),name="relu_hidden")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    conv_hidden_2_out,
                    ksize=[1, 1, 128, 1],#对宽度进行池化
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")

                pooled_outputs.append(pooled) #pooled.shape=[100,20,1,16]

        # Combine all the pooled features
        num_filters_total = 16 * len(filter_sizes)
        self.h_pool2 = tf.concat(pooled_outputs, 3)#[100,20,1,48]

        self.h_pool_flat = tf.reshape(self.h_pool2, [-1, num_filters_total]) #reshape(tensor, to_shape)[,shape]

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], name="b"))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores") #[238000,2]

            self.predictions = tf.argmax(self.scores, axis=1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))

            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
