# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:38:13 2017

@author: pc
"""

import pandas as pd
import os
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer as MLB
from VocBaseClass import VocBaseClass
from NgramFeatureClass import NgramFeatureClass

class LrcClassfierClass(VocBaseClass):
    def _init_(self,train_file_path=None, module_dir=None, tfid=None):
        self._train_file_path = os.path.abspath(train_file_path)
        self._module_dir = os.path.abspath(module_dir)
        if not os.path.exists(self._module_dir):
            os.mkdir(self._module_dir)
        self._feature_extractor = None
        self._model = None
        self._fe = NgramFeatureClass(self._module_dir,tfid)
        
    def train(self):
        print('start train ............')
        self._fe.train(self._train_file_path)
        train_df = pd.read_csv(open(self._train_file_path,'rU'), index_col=None, usecols=['text','label'])
        X = self._fe.gen_feature_batch(train_df['text'].apply(str))
        self._model = LogisticRegression(C=10.0, solver='liblinear')
        label_df = train_df['label'].apply(str).apply(lambda x: x.split('::'))
        self._mlb = MLB()
        train_label = self._mlb.fit_transform(label_df)
        self._model.fit(X,train_label)
        joblib.dump(self._model, self._module_dir + '/voc_oneRestLr.model', compress=1)
        joblib.dump(self.mlb, self._module_dir + '/voc_mlb.model', compress=1)
        
    
    def load_model(self):
        print('start load model ......')
        self._model = joblib.load(self._module_dir + '/voc_oneRestLr.model')
        self.mlb = joblib.load(self._module_dir + '/voc_mlb.model')       
        if self._fe:
            self._fe.load_model()
            
    def classify_batch(self, text_batch):
        print('start classify_batch ............')
        X = self._fe.gen_feature_batch(text_batch)
        score_lst = self._model.predict_proba(X)
        ret = [dict(zip(self._model.classes_, score)) for score in score_lst]
        return ret
    
    
def main():
    voc = LrcClassfierClass('./data/english_v2_group.csv','./data/english/en_group','en')
    voc.train
    voc.load_model()
    voc.predict('./data/english_v2_group_test.csv')
    
    
if __name__ =='__main__':
    main()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    