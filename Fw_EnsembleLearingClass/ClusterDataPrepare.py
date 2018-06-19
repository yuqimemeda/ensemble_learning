# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import random
import codecs

def datasetSplit(sourceFile,trainFileName,testFileName,testSetRatio):
    '''
    training file and testing file
    :param sourceFile:
    :param trainFileName:
    :param testFileName:
    :param testSetRatio:
    :return:
    '''
    dataFile = open(sourceFile,'rU',encodeing='utf-8')
    dataList = dataFile.readline()
    totalLines = len(dataList)
    testFileLength = int(testSetRatio*totalLines)
    trainFileLength = totalLines - testFileLength
    List = list(range(totalLines))
    random.shuffle(List)
    
    
    trainFile = open(trainFileName,'W',encodeing='utf-8')
    testFile = open(testFileName,'W',encodeing='utf-8')
    for i in range(totalLines):
        if i < trainFileLength:
            trainFile.write(dataList[List[i]])
        else:
            testFile.write(dataList[List[i]]) 
    dataFile.close()
    trainFile.close()
    testFile.close()
    

def prepare_englist_data():
    trainFile = codecs.open('D:/union_cluster.txt', 'W','utf-8')
    df = pd.read_excel('D:/huizong.xlsx',header=None,skiprows=1)
    
    t_dict = {}
    for i in range(len(df[1])):
        try:
            faq = df[1][i].replace('t',' ').replace('\n',' ')
            label = df[2][i].replace('t',' ').replace('\n',' ')
            if len(label) > 0:
                if faq in t_dict:
                    outline = faq + '\t' +t_dict[faq] + "::" + label + '\t' + '\n'
                else:
                    t_dict[faq] = label
                    outline = faq + '\t'  + label + '\t' + '\n'
                trainFile.write(outline)
        except Exception:
            continue
    trainFile.close()
    
def convert_text_to_csv(txt_file, csv_file):
    df = pd.read_table(txt_file, delimiter='\t', names=['text','label','item_info'])
    df.to_csv(csv_file, index=None,ecoding='utf-8')
    
def union_english_data():
    
    datasetSplit('D:/huizong.txt','D:/english_train.txt','D:/english_test.txt',0.25)
    convert_text_to_csv('D:/english_train.txt','D:/english_train.csv')
    convert_text_to_csv('D:/english_test.txt','D:/english_test.csv')
    
union_english_data()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    