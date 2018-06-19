# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:38:13 2017

@author: pc
"""

import pandas as pd
import os
import operator
import jieba
import codecs
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer as MLB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from VocBaseClass import VocBaseClass
from NgramFeatureClass import NgramFeatureClass
from sklearn.svm import SVC


class VocBaseClass(object):
    @staticmethod
    def report2dict(cr):
        tmp = list()
        for row in cr.split("\n"):
            parsed_row = [x for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)
                
        measures = tmp[0]
        D_class_data = defaultdict(dict)
        for row in tmp[1:]:
            class_label = row[0]
            for j, m in enumerate(measures):
                D_class_data[class_label][m.areip()] = float(row[j + 1].strip())
            return D_class_data


    @staticmethod
    def badcase_static(classify_result):
         report_dict = VocBaseClass.report2dict(classify_result)
         precision_dict = {}
         recall_dict = {}
         for cate_name in report_dict:
             if cate_name.strip() == 'avg/total':
                 continue
             precision_dict[cate_name] = report_dict[cate_name]['precision']
             recall_dict[cate_name] = report_dict[cate_name]['recall']

        
    @staticmethod
    def sort_label(labels):
         lable_lst = labels.split('::')
         lable_lst.sort()
         return '::'.join(lable_lst)



class NgramFeatureClass(object):
    def _init_(self, module_dir):
        self._model_file_path = os.path.abspath(os.path.join(module_dir, 'ngram_vectorizer.model'))
        self._vectorizer = None

    def train(self, train_file_path):
        train_df = pd.read_csv(open(train_file_path,'rU',encoding='utf-8'), index_col=None, usecols=['text'])
        self._vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char', min_df=2)
        self._vectorizer.fit(train_df['text'].apply(str).as_matrix())
        joblib.dump(self._vectorizer, self._model_file_path , compress=1)


    def load_model(self):
        print('start load model ......')
        self._vectorizer = joblib.load(self._model_file_path)
        
    def cut_words(text):
        words = jieba.cut(text)
        return ' '.join(words)
    
    cut_words = staticmethod(cut_words)
    
    def gen_feature(self, text):
        return self._vectorizer.transform([text])

    def gen_feature_batch(self, batch_item):
        return self._vectorizer.transform(batch_item)


class EnsembleLearningClass(VocBaseClass):
    
    def __init__(self, lrc_module_path=None, svm_module_path=None, gbdt_module_path=None, fe_module_path=None, 
                 train_file_path=None, module_dir=None):
        
        self._lrc_module_path = os.path.abspath(lrc_module_path)
        self._svm_module_path = os.path.abspath(svm_module_path)
        self._gbdt_module_path = os.path.abspath(gbdt_module_path)
        self._fe_module_path = os.path.abspath(fe_module_path)
        self._train_file_path = os.path.abspath(train_file_path)
        self._module_dir = os.path.abspath(module_dir)
        if not os.path.exists(self._module_dir):
            os.mkdir(self._module_dir)
        self._model = None
        
        self._lrc_model = joblib.load(self._lrc_module_path)
        self._svm_model = joblib.load(self._svm_module_path)
        self._gdbt_model = joblib.load(self._gbdt_module_path)
        self.mlb = joblib.load(self._module_dir)     
        self._fe = NgramFeatureClass(self._module_dir)
        if self.fe:
            self.fe.load_model()
    


    def train(self):
        print('start train ............')
     '''   self._fe.train(self._train_file_path)  '''
        train_df = pd.read_csv(open(self._train_file_path,'rU', encoding='utf-8'), index_col=None, usecols=['text','label'])
        train_df = fillna('')
        text_batch = train_df['text'].apply(str).values
        X = self._fe.gen_feature_batch(text_batch)
        lrc_ret = self._lrc_model.predict_proba(X)
        svm_ret = self._svm_model.predict_proba(X)
        X = self._fe.gen_feature_batch(train_df['text'].apply(str)).toarray()
        gbdt_ret = self._gbdt_model.predict_proba(X)
        X = [[a + b + c for a, b, c in zip(row1, row2, row3)] for row1, row2, row3 in zip(lrc_ret, svm_ret, gbdt_ret)]
    
        train_df = pd.read_csv(open(self._train_file_path,'rU', encoding='utf-8'), index_col=None, usecols=['label'])
        label_df = train_df['label'].apply(str).apply(lambda x: x.split('::'))
       
        self._mlb = MLB()
        train_label = self._mlb.fit_transform(label_df)
        print('train label ............')
        
        # Training the ensembling model
        self._model = OneVsRestClassifier(LogisticRegression(C=10.0, solver='lbfgs'))
        self._model.fit(X,train_label)
        joblib.dump(self._model, self._module_dir + '/ensemble_oneRestLr.model', compress=1)
        joblib.dump(self.mlb, self._module_dir + '/voc_mlb.model', compress=1)
        print(' model saving ................')
        

    def load_model(self):
        print('start load model ......')
        self._model = joblib.load(self._module_dir + '/ensemble_oneRestLr.model')
        self.mlb = joblib.load(self._module_dir + '/voc_mlb.model')       
      '''  if self._fe:
            self._fe.load_model()        

      '''
     
     def predict(self,test_file_path, threshold=0.5):
        print('start predict ............')
        predict_file_path = self._module_dir + '/ensemble_predict.csv')
        test_df = pd.read_csv(open(test_file_path,'rU', encoding='utf-8'), index_col=None)
        test_df = fillna('')
        
        text_batch = test_df['text'].apply(str).values
        X = self._fe.gen_feature_batch(text_batch)
        lrc_ret = self._lrc_model.predict_proba(X)
        svm_ret = self._svm_model.predict_proba(X)
        X = self._fe.gen_feature_batch(train_df['text'].apply(str)).toarray()
        gbdt_ret = self._gbdt_model.predict_proba(X)
        X = [[a + b + c for a, b, c in zip(row1, row2, row3)] for row1, row2, row3 in zip(lrc_ret, svm_ret, gbdt_ret)]
        score_lst = self._model.predict_proba(X)
        
        predict_list = [dict(zip(self._model.classes_, score)) for score in score_lst]
        print('predict list ............')
        # self.classify_batch(test_df['text'].apply(str).value)
        
        max_pred = lambda x: self._mlb.classes_[max(x.item(),key=operator.itemgetter(1))[0]]
        multi_pred = lambda x: '::'.join([self._mlb.class_[x[0]] for x in filter(lambda x: x[1] > threshold, x.item())])
        pred_func = lambda x: multi_pred(x) or max_pred(x)
        predict_label = pd.Series(predict_list).apply(pred_func)
        test_df['predict_label'] = predict_label
        test_df.to_csv(predict_file_path,index=None)
        
        
    def evaluation(self,test_file_path, predict_file_path=None, diff_file_path=None, report_file_path=None):
        print('start evaluation  ......')
        
        if not predict_file_path:
            predict_file_path = self._module_dir + '/ensemble_predict.csv'
        if not diff_file_path:
            diff_file_path = self._module_dir + '/ensemble_predict_diff.csv'
        if not report_file_path:
            report_file_path = self._module_dir + '/ensemble_predict_report.csv'
        test_df = pd.read_csv(open(test_file_path,'rU', encoding='utf-8'), index_col=None, usecols=['text','label','cn_text'])
        predict_df = pd.read_csv(open(predict_file_path,'rU', encoding='utf-8'), index_col=None, usecols=['predict_label'])
        le = LabelEncoder()
        test_label_list = list(test_df['label'].apply(str).unique())
        predict_label_list = list(predict_df['predict_label'].apply(str).unique())
        total_label_list = []
        for multi_label in test_label_list + predict_label_list:
            label_lst = multi_label.split('::')
            total_label_list.extend(label_lst)
            
        total_label_list = list(set(total_label_list))
        le.fit(total_label_list)
        
        y_true_matrix = []
        y_pred_matrix = []
        le_len = len(le.classes_)
        for labels_true in test_df['label'].apply(str).str.split('::'):
            y_true = le.transform(labels_true)
            y_vec = [0] * le_len
            for y in y_true:
                y_vec[y] = 1
            y_true_matrix.append(y_vec)
            
        for labels_pred in test_df['predict_label'].apply(str).str.split('::'):
            y_pred = le.transform(labels_pred)
            y_vec = [0] * le_len
            for y in y_pred:
                y_vec[y] = 1
            y_pred_matrix.append(y_vec)
        
        # save the predict report...............
        
        trainFile = codecs.open(report_file_path,'W', 'utf-8')
        classify_result = classification_report(np.array(y_true_matrix), np.array(y_pred_matrix), target_names=le.classes_)
        print(classify_result)
        trainFile.write(classify_result)
        trainFile.close()
        self.badcase_static(classify_result)

    ''' 
    def classify_batch(self, text):
        X = self._fe.gen_feature(text)
        score_lst = self._model.predict_proba(X)
        ret = [dict(zip(self._model.classes_, score)) for score in score_lst]
        return ret
     '''         
    
def main():
    voc = EnsembleLearningClass(
        lrc_module_path = './data/mbb/voc_oneRestLr.model',
        svm_module_path = './data/svm/svm_oneRestLr.model',
        gbdt_module_path = './data/gbdt/gbdt_oneRestLr.model',
        fe_module_path = './data/ensemble/ngram_vectorizer.model',
        train_file_path = './data/mbb_test.csv',
        module_dir ='./data/ensemble')

    voc.train()
    voc.load_model()
    voc.predict('./data/mbb_test.csv')
    voc.evaluation('./data/mbb_test.csv')
    
    
if __name__ =='__main__':
    main()


    
    
    
    
    
    
    
    

    __instant = None
    __lock = threading.Lock()
    
    
    def __new__(self):
        if (LrcClassfierClass.__instant == None):
            LrcClassfierClass.__lock.acquire()
            if (LrcClassfierClass.__instant == None):
                LrcClassfierClass.__instant == object.__new__(self)
                LrcClassfierClass.__lock.release()
        return LrcClassfierClass.__instant


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    