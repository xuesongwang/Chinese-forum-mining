# -*- coding: utf-8 -*-
__author__ = 'Zonglin Di'

import json
import time
import codecs

def dict2json(dict,url):
    today = time.strftime('%Y%m%d%H%M%S', time.localtime())
    fileName = '../DataSource/Output/result.json'
    with codecs.open(fileName, 'a', encoding='utf8') as fp:
        line = '{"post":'
        line = line +  json.dumps(dict["post"], ensure_ascii=False) + ',replys:['
        for reply in dict["reply"]:
            line = line + json.dumps(reply, ensure_ascii=False)
        line = url+'\t'+line + ']'+'\n'
        fp.write(line)