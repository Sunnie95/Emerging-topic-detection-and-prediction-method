# -*- coding: utf-8 -*-
######################################
# 1、连接Mysql数据库,读取2013-2016年互联网新闻数据
# 2、存储原始数据，dump_news()
# 3、中文分词并存储处理后数据，seg_news()
#####################################

import re
import codecs
import MySQLdb
import sys
import os
import time
import jieba
import logging
import io
jieba.load_userdict("./dictionary/1.financial_complete.txt")
jieba.load_userdict("./dictionary/2.stocks_complete.txt")
jieba.load_userdict("./dictionary/3.2014stock.txt")
jieba.load_userdict("./dictionary/4.2016stock.txt")
jieba.load_userdict("./dictionary/5.Astock_short.txt")

#连接数据库
db = MySQLdb.connect(host=" ", user=" " ,passwd=" ",db=" ",port=  ,charset=" ")
cursor = db.cursor()

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath,'r',encoding= 'utf-8').readlines()]
    return stopwords

def removePunc(itext):
    pattern = r'[\d+\s+\.\!\/_,?=$%^)*(+\"\']+|[+——！:；>    ，。？、~@#￥%……&*（）]'
    newstr = re.sub(pattern,'',itext)
    r1 = re.sub("[<>]","",newstr)
    r2 = re.sub("[a-zA-Z0-9]","",r1)
    return r2

def matchc(itext):
    pattern = r'[\u4e00-\u9fa5]'
    newstr = re.sub(pattern,'',itext)

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

class Vocabulary:
    def dump_news(self):
        #读取新闻的标题、关键词和日期
        sql = "select title,keywords,day_f from ned_stocknews"
        cursor.execute(sql)
        rows = cursor.fetchall()
        count = 0
        dumpfile = "./dumpnews.txt"
        for i,row in enumerate(rows):
            title = row[0]
            keywords = row[1]
            keywords = "#".join(keywords.split(','))
            with open(dumpfile, "a") as f:
                count += 1
                f.write(str(count) + ':' + title +'。'+ keywords +'\n')
        print ("the number of news:",count)
        return dumpfile

    def seg_news(self,dfile):
        filelines = file_len(dfile)
        stopwords = stopwordslist('./dictionary/stopwords_2017.txt')  # 这里加载停用词的路径
        texts_num = 0
        segfile = "./segnews.txt"
        with io.open(segfile, "w", encoding='utf-8') as output:
            with io.open(dfile,"r",encoding='utf-8') as content:
                for line in content:
                    if line:
                        outstr = ''
                        seg_list = jieba.cut(line)
                        for word in seg_list:
                            if word not in stopwords:
                                newword = removePunc(word)
                                if len(newword)>1:
                                    outstr += newword.strip()
                                    outstr += ' '
                        output.write(outstr+'\n')
                    texts_num += 1
                if texts_num % 10 == 0:
                    print("已完成分詞行数：" ,(texts_num/float(filelines)*100),"...")
        return segfile

def run():
    voca = Vocabulary()
    dumpfile = voca.dump_news()
    segfile = voca.seg_news(dumpfile)       

if __name__ == "__main__":
    run()
