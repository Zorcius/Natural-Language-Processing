# -*- coding: utf-8 -*-
#coding=gbk
#coding=utf-8

import numpy as np
import re
import os
import word2vec_helpers
import time
import pickle
import jieba
import alphachange as alph
import langconv as converter
import random

def load_data_and_labels(input_text_file, input_label_file, num_labels):
    x_text = read_and_clean_zh_file(input_text_file)
    y = None if not os.path.exists(input_label_file) else [int(item) for item in list(open(input_label_file, "r").readlines())]
    return (x_text, y)

# 处理数据
def load_positive_negative_data_files(positive_data_file, negative_data_file, cut, stop_words_list_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = read_and_clean_zh_file(positive_data_file, cut_words=cut,
                                               stop_words_list_file=stop_words_list_file,output_cleaned_file="C:\\Zorcius\\filtering\\clean_pos.txt")
    negative_examples = read_and_clean_zh_file(negative_data_file, cut_words=cut,
                                               stop_words_list_file=stop_words_list_file,output_cleaned_file="C:\\Zorcius\\filtering\\clean_neg.txt")
    # Combine data
    x_text = positive_examples + negative_examples
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples] #正样本标为1
    negative_labels = [[1, 0] for _ in negative_examples] #负样本标为0
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def loadClassData(filename):
    dataList  = []
    for line in open('./data/'+filename,'r',encoding='utf-8').readlines():#读取分类序列
        dataList.append(line.strip())
    return dataList

def loadTrainData(filename):
    dataList  = []
    for line in open('./data/'+filename,'r',encoding='utf-8').readlines():
        dataList.append(line.strip())
    return dataList

def load_training_data_files(training_data_file):
    x_text = read_and_clean_zh_file(training_data_file)
    y = np.array(loadClassData("classLabel.txt"),dtype=int)
    return [x_text, y]


# 切分词
def cutWords(msg, stopWords):
    seg_list = jieba.cut(msg, cut_all=False)
    # key_list = jieba.analyse.extract_tags(msg,20) #get keywords
    leftWords = []
    for word in seg_list:
        if word not in stopWords:
            leftWords.append(word)
    return leftWords
def preProcess(uStr):
    ustring = uStr.replace(' ', '')
    # ret=string2List(ustring.decode('utf-8'))
    ret = alph.string2List(ustring)
    msg = ''
    for key in ret:
        key = converter.Converter('zh-hans').convert(key)
        msg += key
        # ustring =   msg.encode('utf-8')
    ustring = msg
    ustring = ustring.replace('x元', '价钱')
    ustring = ustring.replace('x日', '日期')
    ustring = ustring.replace('x折', '打折')
    ustring = ustring.replace('www', '网站')

    return ustring

# 填充句子
def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    padded = []
    max_sentence_length = padding_sentence_length
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[-max_sentence_length:]
            padded.append(sentence)
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
            padded.append(sentence)
    return padded


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    # Generate a batch iterator for a dataset

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1 #根据batch_size来分割整个data_size
    for epoch in range(num_epochs):
        if shuffle:
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx: end_idx]

def mkdir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def seperate_line(line):
    return ''.join([word + ' ' for word in line])


# 将文件中每个字用空格分隔开
def read_and_clean_zh_file(input_file, output_cleaned_file=None, cut_words=False, stop_words_list_file=None):
    #lines = list(open(input_file).readlines())
    lines = list(open(input_file, encoding='UTF-8').readlines())
    #lines = list(open(input_file, encoding='gbk').readlines())

    if cut_words and stop_words_list_file is not None:#选择切分词
        stop_words_list = list(open(stop_words_list_file,encoding='utf-8').readlines())
        lines = [clean_str(cutWords(line, stop_words_list)) for line in lines]
    else:
        #lines = [clean_str(seperate_line(line)) for line in lines]#去除标点符号
        #lines = [clean_str_with_punctuation_kept(seperate_line(shuffle_string(line))) for line in lines]#打乱词序->空格分句->保留问号和叹号
        lines = [clean_str_with_punctuation_kept(seperate_line(shuffle_string_only20(line))) for line in lines]
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w') as f:
        #with open(output_cleaned_file, 'w',encoding='gbk') as f:
            for line in lines:
                f.write((line + '\n'))
    return lines


# 去除标点符号
def clean_str(string):
    # Remove punctuation
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
def clean_str_with_punctuation_kept(string):
    '''
    去掉逗号、句号，但是保留一些带有语气转换的标点符号，譬如感叹号、问号
    '''
    string = re.sub(r"[a-zA-Z]","",string)#替换所有的字母
    string = re.sub(r"[0-9\\.\\*\-]","",string)#替换所有的数字，星号，破折号
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def drop(string):
    index = [i for i in np.arange(len(string))] #生成该句子长度的索引值
    drop_num = len(string)-20
    drop_index = random.sample(index, drop_num) #从生成的索引值中不重复地抽出len-20个索引,
    kept_string = [string[i] for i in index if i not in drop_index]#删除这些索引对应的字
    return "".join(kept_string)

def shuffle_string(string):
    '''
   该函数用在padding之前，因为一句话的最大长度设定为20，所以在截断之前，
   先打乱词顺序来减少信息的丢失。
   对于长度大于20的句子，不必打乱词顺序，采用随机drop掉(len-20)个字的方法
   函数返回一个string
   '''
    string_len = len(string)
    shuffle_indices = np.random.permutation(np.arange(string_len))
    if string_len > 20:
        return drop(string)

    return "".join([string[index] for index in shuffle_indices])

def shuffle_string_only20(string):
    '''
    只针对长度大于20的句子进行打乱词序+随机drop
    '''
    string_len = len(string)
    if string_len > 20:
        return drop(string)
    return string

def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f)


def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict
