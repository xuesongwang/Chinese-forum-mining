# -*- coding: utf-8 -*-
__author__ = 'Xuesong Wang'

import pickle
import sys
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os,sys
from CNN.data_helper import *
import re
import copy


"""decoding chinese characters"""
reload(sys)
sys.setdefaultencoding('utf-8')


def Readfile(validname,rubbishname,size,rubbishsize):
    """a copy of readfile to build separately from CNN"""
    valid = pd.read_csv(validname, encoding='utf-8')
    rubbish = pd.read_csv(rubbishname,encoding='utf-8')
    x_valid = valid['content'].values
    y_valid = valid['label'].values
    y_valid[:]= 'valid'
    x_rubbish = rubbish['content'].values
    y_rubbish = rubbish['label'].values
    x=np.concatenate((x_valid[0:size],x_rubbish[0:rubbishsize]))
    y = np.concatenate((y_valid[0:size],y_rubbish[0:rubbishsize]))
    shuffle_indices = np.random.permutation(np.arange(len(x)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    return x,y


def FeaureExtraction(contents,vocab_processor = None):
    """ pre-processing input data ,removing meaningless signs"""
    seg_list = []
    coder = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）“”]+"
    for content in contents:
        content = str(content).encode("utf-8")
        string = re.split(coder.decode("utf8"),content)
        for substr in string :
            if substr:
                s = filter(str.isdigit, substr)
                if s:
                    seg_list.append(copy.deepcopy(s))
                seg_list.append(substr.replace(s,''))
    """ remove redundant records """
    seg_list = list(set(seg_list))
    contentlist =[]
    for l in seg_list:
        l = str(l).encode("utf-8")
        word = list(jb.cut(l))
        seg = ''
        for x in word:
            seg += x.strip() + ' '
        contentlist.append(seg)
    """ pad each sentence to the same length and map each word to an id"""
    if vocab_processor:
        x = np.array(list(vocab_processor.transform(contentlist)))
    else:
        max_document_length = max([len(seg.split(' ')) for seg in contentlist])
        logging.info('The maximum length of all sentences in training set: {}'.format(max_document_length))
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(seg_list)))
    return x,vocab_processor,seg_list


def Predict(alldata):
    """predict whether alldata is valid """
    with open('../SVM/data/binary_vocab.pkl', 'r') as f:
        vocab_processor = pickle.load(f)
    with open('../SVM/binarymodel.pkl', 'r') as f:
        clf = joblib.load(f)
    x_test, vocab_processor,contentlist = FeaureExtraction(alldata, vocab_processor)
    y_pred = clf.predict(x_test)
    return y_pred,contentlist

if __name__ == '__main__':
    x,y =Readfile('../DataSource/valid.csv','../DataSource/rubbish.csv',4000,3000)
    x_train, x_test, y_, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # a = Predict(x_train)
    x_, vocab_processor = FeaureExtraction(x_train)
    x_test,vocab_processor = FeaureExtraction(x_test,vocab_processor)
    clf = svm.SVC()
    clf.fit(x_, y_)
    print clf.score(x_test,y_test)
    cm = confusion_matrix(y_test,clf.predict(x_test),labels=['rubbish','valid'])
    print cm
    with open('../SVM/data/binary_vocab.pkl', 'w') as f:  # open file with write-mode
        pickle.dump(vocab_processor, f)
    with open('../SVM/binarymodel.pkl', 'w') as f:
        joblib.dump(clf, f)

