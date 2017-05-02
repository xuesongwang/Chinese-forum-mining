# -*- coding: utf-8 -*-
__author__ = 'Xuesong Wang'

import uniout
import os
import sys
import json
import logging
from data_helper import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)
"""decoding chinese characters"""
reload(sys)
sys.setdefaultencoding('utf-8')


s = [u'的分数高如果认购二哥让他退给我', u'从这条新闻中你得到了什么教训？\n\r\n在网上看个新闻，大概内容是：', u'啊实打实as分噶都是gas豆腐干']
vocab_path = "vocab.pickle"
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test, vocab_processor = wordSeg(s, vocab_processor=vocab_processor)
print x_test