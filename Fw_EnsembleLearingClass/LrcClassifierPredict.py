# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:38:13 2017

@author: pc
"""

import pandas as pd
import os
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import jieba
import threading

class NgramFeatureClass(object):
    def _init_(self, module_dir):
        self._model_file_path = os.path.abspath(os.path.join(module_dir, 'ngram_vectorizer.model'))
        self._vectorizer = TfidfVectorizer(ngram_range=(1,1), analyzer='word', min_df=2)

    def load_model(self):
        print('start load model ......')
        self._vectorizer = joblib.load(self._module_file_path)
        
    def cut_words(text)
        words = jieba.cut(text)
        return ' '.join(words)
    
    cut_words = staticmethod(cut_words)
    
    def gen_feature(self, text):
        return self._vectorizer.transform([text])

    def gen_feature_batch(self, batch_item):
        return self._vectorizer.transform(batch_item)


class LrcClassfierClass(object):

    __instant = None
    __lock = threading.Lock()
    
    
    def __new__(self):
        if (LrcClassfierClass.__instant == None):
            LrcClassfierClass.__lock.acquire()
            if (LrcClassfierClass.__instant == None):
                LrcClassfierClass.__instant == object.__new__(self)
                LrcClassfierClass.__lock.release()
        return LrcClassfierClass.__instant
    
    
    def __init__(self, module_dir='./data/english'):
        self._module_dir = os.path.abspath(module_dir)
        if not os.path.exists(self._module_dir):
            os.mkdir(self._module_dir)
        self._model = None
        self._fe = NgramFeatureClass(self._module_dir)
        self._model = joblib.load(self._module_dir + '/voc_oneRestLr.model')
        self.mlb = joblib.load(self._module_dir + '/voc_mlb.model')      
        if self.fe:
            self.fe.load_model()
            
 
    def classify_batch(self, text):
        X = self._fe.gen_feature(text)
        score_lst = self._model.predict_proba(X)
        ret = [dict(zip(self._model.classes_, score)) for score in score_lst]
        return ret


    def predict(self,text, threshold=0.5):
        predict_list = self.classify_batch(text)
        max_pred = lambda x: self._mlb.classes_[max(x.item(),key=operator.itemgetter(1))[0]]
        multi_pred = lambda x: '::'.join([self._mlb.class_[x[0]] for x in filter(lambda x: x[1] > threshold, x.item())])
        pred_func = lambda x: multi_pred(x) or max_pred(x)
        predict_label = pd.Series(predict_list).apply(pred_func)
        return predict_label
               
    
def main():
    voc = LrcClassfierClass()
    predict = voc.predict('Yes, the signal is stable!')
    print(predict[0])
    print(type(predict[0]))

    
    
if __name__ =='__main__':
    main()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    