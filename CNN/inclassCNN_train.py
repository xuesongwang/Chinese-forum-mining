# -*- coding: utf-8 -*-
__author__ = 'Xuesong Wang'


import sys
from data_helper import *
from sklearn.model_selection import train_test_split
import json
import logging
import tensorflow as tf
from text_cnn import TextCNN
import time
import os


if __name__ == '__main__':
    """ Step 0: reload coding systems to display correct Chinese characters"""
    reload(sys)
    sys.setdefaultencoding('utf-8')

    """ Step 1: read data ,model parameters """
    # data = readFile('./data/txt/')
    data = pd.read_csv('../DataSource/valid.csv',encoding="utf-8")
    parameter_file = '../CNN/data/parameters.json'
    params = json.loads(open(parameter_file).read())

    """ Step 2: Select top 4000 shuffled samples as total data sets and ...
     build word embedding vocabulary therein """
    shuffle_indice = range(0,len(data))
    np.random.shuffle(shuffle_indice)
    x,vocab_processor = wordSeg(data['content'].values[shuffle_indice],initial_vocab=4000)
    # seg_list = pd.read_csv('./data/feature.csv',encoding='utf-8',header=None).values[:, 0]
    # x, vectorizer, transformer = tfIdf(seg_list)

    """ Step 3: map labels into one hot vector """
    y,enc,enc2 = labelEncoding(data['label'].values[shuffle_indice[0:4000]])

    """ Step 4: split into training and testing"""
    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    """ Step 5: shuffle train sets and split them into train and dev sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y_)))
    x_shuffled = x_[shuffle_indices]
    y_shuffled = y_[shuffle_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    """Step 6: build a graph and cnn object"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size = len(vocab_processor.vocabulary_),
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'],
                l2_reg_lambda=params['l2_reg_lambda'],
                labelencoder=enc)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = "trained_model_" + timestamp
            # decide path to save model
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch,i):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params['dropout_keep_prob']}
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter('./CNN/graphs/train',sess.graph)
                summary,_, step, loss, acc = sess.run([merged,train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
                writer.add_summary(summary,i)
                writer.close()
                # the following run without building a graph for tensorboard
                #  _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch, i=None):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter('../CNN/graphs/valid', sess.graph)
                summary,step, loss, acc, num_correct,confusion = sess.run([merged,global_step, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.confusion], feed_dict)
                if i:
                    writer.add_summary(summary,i)
                # step, loss, acc, num_correct, confusion = sess.run(
                #     [global_step, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.confusion], feed_dict)
                return num_correct,confusion

            # Save the word_to_id map since predictLabel.py needs it
            # vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
            # transformer.save(os.path.join(out_dir, "vocab.pickle"))

            # Initialize the graph
            sess.run(tf.global_variables_initializer())

            # Training starts here
            # split training data into batch
            train_batches = batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
            best_accuracy, best_at_step = 0, 0

            """Step 7: train the cnn model with x_train and y_train (batch by batch)"""
            for batchindex, train_batch in enumerate(train_batches):
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch, batchindex)
                current_step = tf.train.global_step(sess, global_step)
                """Step 7.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct,confusion_matrix = dev_step(x_dev_batch, y_dev_batch,current_step)
                        total_dev_correct += num_dev_correct

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                    """Step 7.2: save the model if it is the best based on accuracy of the dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

            """Step 8: predict x_test (batch by batch)"""
            test_batches = batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            correctlist = np.zeros(len(enc.classes_),)
            incorrectlist = np.zeros(len(enc.classes_),)
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct, confusion_matrix= dev_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct
                for i in range(0, confusion_matrix.shape[0]):
                    correctlist[i] += confusion_matrix[i,i]
                    incorrectlist[i] += np.sum(confusion_matrix[i, :], axis=0) - confusion_matrix[i,i]
            df = DataFrame([correctlist, incorrectlist], index=['correct', 'incorrect'], columns=enc.classes_)
            df.to_csv('data/result.csv')
            # draw confusion matrix
            confusionmatri_show(df)
            test_accuracy = float(total_test_correct) / len(y_test)
            logging.critical('Accuracy on test set is {} '.format(test_accuracy))
            logging.critical('The training is complete')

