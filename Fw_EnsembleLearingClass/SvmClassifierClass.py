# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:38:13 2017

@author: pc
"""

import pandas as pd
import os
import operator
import numpy as np
import codecs

from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer as MLB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from VocBaseClass import VocBaseClass
from NgramFeatureClass import NgramFeatureClass
from sklearn.svm import SVC

class SvmClassfierClass(VocBaseClass):
    def _init_(self,train_file_path=None, module_dir=None):
        self._train_file_path = os.path.abspath(train_file_path)
        self._module_dir = os.path.abspath(module_dir)
        if not os.path.exists(self._module_dir):
            os.mkdir(self._module_dir)
        self._model = None
        self._fe = NgramFeatureClass(self._module_dir,tfid)
        
    def train(self):
        print('start train ............')
        self._fe.train(self._train_file_path)
        train_df = pd.read_csv(open(self._train_file_path,'rU',encoding='utf-8'), index_col=None, usecols=['text','label'])
        X = self._fe.gen_feature_batch(train_df['text'].apply(str))
        self._model = SVC(kernel='linear', C=100000, probability=True, random_state=0)
        label_df = train_df['label'].apply(str).apply(lambda x: x.split('::'))
        self._mlb = MLB()
        train_label = self._mlb.fit_transform(label_df)
        self._model.fit(X,train_label)
        joblib.dump(self._model, self._module_dir + '/svm_oneRestLr.model', compress=1)
        joblib.dump(self.mlb, self._module_dir + '/svm_mlb.model', compress=1)
        
    
    def load_model(self):
        print('start load model ......')
        self._model = joblib.load(self._module_dir + '/svm_oneRestLr.model')
        self.mlb = joblib.load(self._module_dir + '/svm_mlb.model')       
        if self._fe:
            self._fe.load_model()
            
    def classify_batch(self, text_batch):
        print('start classify_batch ............')
        X = self._fe.gen_feature_batch(text_batch)
        score_lst = self._model.predict_proba(X)
        ret = [dict(zip(self._model.classes_, score)) for score in score_lst]
        return ret
    
    
     def predict(self,test_file_path, threshold=0.5):
        print('start predict ............')
        predict_file_path = self._module_dir + '/svm_predict.csv')
        test_df = pd.read_csv(open(test_file_path,'rU'), index_col=None)
        test_df = fillna('')
        predict_list = self.classify_batch(test_df['text'].apply(str).value)
        max_pred = lambda x: self._mlb.classes_[max(x.item(),key=operator.itemgetter(1))[0]]
        multi_pred = lambda x: '::'.join([self._mlb.class_[x[0]] for x in filter(lambda x: x[1] > threshold, x.item())])
        pred_func = lambda x: multi_pred(x) or max_pred(x)
        predict_label = pd.Series(predict_list).apply(pred_func)
        test_df['predict_label'] = predict_label
        test_df.to_csv(predict_file_path,index=None)
     
    
    
def main():
    voc = SvmClassfierClass('./data/mbb_train.csv','./data/svm')
    voc.train()
    voc.load_model()
    voc.predict('./data/mbb_test.csv')
    
    
if __name__ =='__main__':
    main()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    