# -*- coding: utf-8 -*-
__author__ = 'Xuesong Wang'

import os
import sys
import json
import logging
from CNN.data_helper import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
import codecs
import copy
from SVM import BinaryclassSVM


logging.getLogger().setLevel(logging.INFO)
"""decoding chinese characters"""
reload(sys)
sys.setdefaultencoding('utf-8')


def predict_unseen_data(onewebpage,checkpoint_dir= '../CNN/trained_model_1493639329'):
    """ Step 0: load trained model and parameters """
    params = json.loads(open('../CNN/data/parameters.json').read())
    if not checkpoint_dir.endswith('/'):
        checkpoint_dir += '/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
    logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

    """Step 1: load data for prediction"""
    # labels.json was saved during training, and it has to be loaded during prediction
    with open('../CNN/data/lableEncoders.pkl', 'r') as f:
        [enc,enc2] = pickle.load(f)

    logging.info('The number of post and reply: {}'.format(len(onewebpage)))

    vocab_path = "../CNN/data/vocab.pickle"
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    data = {'content': [], 'author': [], 'title': [], 'publish_date': []}
    Data = {"post": data, "reply": []}
    for postid,onepost in enumerate(onewebpage):

        """Step 2:Filter spam information by SVM model"""
        validchecker,contentlist = BinaryclassSVM.Predict(onepost)
        validlist = []
        for j in range(0,len(contentlist)) :
            if validchecker[j]== 'valid':
                validlist.append(contentlist[j])
        x_test, vocab_processor = wordSeg(validlist,vocab_processor=vocab_processor)

        """Step 3:Load CNN model"""
        if not checkpoint_dir.endswith('/'):
            checkpoint_dir += '/'
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
        logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

        """Step 4: compute the predictions"""
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)

            with sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                input_x = graph.get_operation_by_name("input_x").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # batches = batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
                all_predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})

        labelpredict =[]

        for i in all_predictions:
            labelpredict.append(enc.classes_[i])
        """Step 4: Save into a dict """
        data = {'content': [], 'author': [], 'title': [], 'publish_date': []}
        content =''
        date =[]
        author =[]
        for index,one in enumerate(labelpredict):
            if one =='author':
                author.append(validlist[index])
            if one =='publish_date':
                date.append(validlist[index])
            else:
                content += validlist[index]

        data['content'].append(content)
        """Step 5: filter potential publish date by designing some rule"""
        minlen =50
        d_min =''
        for d in date:
            if d.isdigit:
                if len(d)<=5:
                    data['publish_date'].append(d)
                    break
                elif len(d)<minlen:
                    d_min=copy.deepcopy(d)
                    minlen = len(d)
        if data['publish_date'] and len(d)>5:
            data['publish_date'].append(d_min)

        """Step 6:filter potential publish date by designing some rule"""
        min_length = 50
        a_opti = ''
        for x in author:
            if len(x) < min_length:
                a_opti = copy.deepcopy(x)
                min_length = len(x)
        data['author'].append(a_opti)
        if postid ==0:
            Data["post"]=copy.deepcopy(data)
        else:
            Data["reply"].append(copy.deepcopy(data))
    return Data

if __name__ == '__main__':
    # python3 predictLabel.py ./trained_model_1478649295/ ./data/small_samples.json

    # datalist =[u'鸡蛋君', u'2017-3-20', u'要春天啦', u'时间总是过的飞快 冬天的大雪']
    data_dict = predict_unseen_data(onepagelist,'./trained_model_1492693456/')


