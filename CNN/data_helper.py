# -*- coding: utf-8 -*-
__author__ = 'Xuesong Wang'

import pandas as pd
import numpy as np
import jieba as jb
from pandas import DataFrame,Series
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import logging
from tensorflow.contrib import learn
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle


def GetFileList(dir,fileList):
    """get file list from a directory"""
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList


def readFile(filename,rubbishfile):
    """read date and label from files"""
    filelist =[]
    filelist = GetFileList(filename,filelist)
    filelen = len(filelist)
    contentlist =[]
    labellist =[]
    content =''
    s = [u'author', u'content', u'title', u'publish_date']
    for filerecord in filelist:
        if os.path.getsize(filerecord):
            df = pd.read_table(filerecord,sep='\n',header=None, encoding="utf-8",).values[:,0]
            for obj in df:
                if obj not in s:
                    content = content+obj
                else:
                    contentlist.append(content)
                    content = ''
                    labellist.append(obj)
    recordlen = len(contentlist)
    df = DataFrame({'content': contentlist, 'label': labellist})
    df.to_csv('../DataSource/Output/valid.csv',encoding="utf-8")
    contentlist = []
    filelist = []
    labellist = []
    rubbishlist = GetFileList(rubbishfile,filelist)
    for i,filerecord in enumerate(rubbishlist):
        if os.path.getsize(filerecord):
            df = pd.read_table(filerecord,sep='\n',encoding="utf-8",).values[:,0]
            for obj in df:
                if obj:
                    contentlist.append(obj)
                    labellist.append('rubbish')
    df = DataFrame({'content': contentlist, 'label': labellist})
    df.to_csv('../DataSource/Output/rubbish.csv', encoding="utf-8")

    print 'total file:%d , no of records:%d'%(filelen,len(contentlist))
    return contentlist,labellist


def wordSeg(contents,initial_vocab=None,vocab_processor=None):
    """first time just use vocab_recordsize to decide how many words to build a vocabulary """
    if initial_vocab:
        seg_list =[]
        for content in contents[0:initial_vocab]:
            content = str(content).encode("utf-8")
            word = list(jb.cut(content))
            seg = ''
            # split words and catch them with whitespace
            for x in word:
                seg += x.strip() + ' '
            seg_list.append(seg)
        """pad each sentence to the same length and map each word to an id"""
        max_document_length = max([len(seg.split(' ')) for seg in seg_list])
        logging.info('The maximum length of all sentences in training set: {}'.format(max_document_length))
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(seg_list)))
        # vocab_processor.save(os.path.join("../CNN/data/vocab.pickle"))

    # """only need to cut and transform testing set"""
    else:
        seg_list =[]
        for content in contents:
            content = str(content).encode("utf-8")
            word = list(jb.cut(content))
            seg = ''
            for x in word:
                seg += x.strip() + ' '
            seg_list.append(seg)
        x = np.array(list(vocab_processor.transform(seg_list)))
    return x, vocab_processor


def labelEncoding(y_raw):
    enc = LabelEncoder()
    y_chategory = enc.fit_transform(y_raw)
    y_re = y_chategory.reshape((len(y_chategory), 1))
    enc2 = OneHotEncoder()
    y = enc2.fit_transform(y_re).toarray()
    """ save encoders to the directory """
    with open('../CNN/data/lableEncoders.pkl', 'w') as f:  # open file with write-mode
        pickle.dump([enc,enc2], f)
    return y,enc,enc2


def confusionmatri_show(df):
    acc = (df.ix['correct'] / (df.ix['correct'] + df.ix['incorrect'])).values
    acc = np.nan_to_num(acc)
    labellist = df.columns
    correct = df.ix['correct'].values
    incorrect = df.ix['incorrect'].values
    # to give a nice format to show confusion matrix
    print "{0:<13}{1:^10}{2:^10}{3:^10}".format(" ","correct","incorrect","accuracy")
    for i in range(0,acc.shape[0]):
        print "{0:<13}{1:^10}{2:^10}{3:^10}".format(labellist[i],correct[i],incorrect[i],acc[i])


def tfIdf(seg_list):
    """another way to encode word features"""
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(seg_list))
    weight = tfidf.toarray()
    return weight,vectorizer,transformer


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(np.ceil(data_size / batch_size))

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    content1,label1 =readFile('data/DataSet/_validate','data/DataSet/_rubbish')
    content2,label2 = readFile('data/2nd DataSet/_validate','data/2nd DataSet/_rubbish')
    content = content1+content2
    label = label1+ label2
    data = {'content':content,'label':label}
    df = DataFrame(data)
    df.to_csv('data/data.csv',index=False,encoding="utf-8")