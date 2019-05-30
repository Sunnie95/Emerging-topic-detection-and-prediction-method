# -*-coding:utf-8 -*-
##########################################
#数据处理，转化为id,计算词频
##########################################

import codecs
import matplotlib.pyplot as plt
import pandas as pd

class Vocabulary:
    def __init__(self, excluds_stopwords=False):
        self.vocas = []        # id to word
        self.vocas_id = dict() # word to id
        self.docfreq = []      # id to document frequency
        self.excluds_stopwords = excluds_stopwords

    def term_to_id(self, term):
        try:
            term_id = self.vocas_id[term]
        except:
            term_id = len(self.vocas)
            self.vocas_id[term] = term_id
            self.vocas.append(term)
            self.docfreq.append(0)
        return term_id

    def doc_to_ids(self, doc):
        l = []
        words = dict()
        for term in doc.split():
            id = self.term_to_id(term)
            if id != None:
                l.append(id)
                if not id in words:
                    words[id] = 1
                    self.docfreq[id] += 1 # It counts in how many documents a word appears. If it appears in only a few, remove it from the vocabulary using cut_low_freq()
        if "close" in dir(doc): doc.close()
        return l

    def choose_freq(self, docs, threshold1,threshold2):
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq >= threshold1 and freq <= threshold2:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq
        #return new_vocas,new_docfreq

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map: new_doc.append(conv_map[id])
            return new_doc
        return [conv(doc) for doc in docs]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)
