# -*- coding: utf-8 -*- 
########################################
# HDP的实现
# 1、setargs(),调参，得到使模型复杂度最小的参数组合
# 2、newrun(),计算并保存最终结果
###########################################


import logging
import numpy
import math
import codecs
import prechinese
from numpy.random import choice
import pandas as pd
import csv

segfile = "./segnews.txt"
K0 = 50#初始主题数
arg_iter = 50#调参中的迭代次数
min_iter = 298#最少迭代次数
max_iter = 1000#求解HDP的最大迭代次数
absper = 1#控制迭代的per差值

class HDP_gibbs_sampling:
    def __init__(self, K0=None, alpha=None, beta=None,gamma=None, docs= None, V= None):
        self.iteration = 0
        self.maxnn = 1
        self.alss=[] # an array for keep the stirling(N,1:N) for saving time consumming
        self.K = K0  # initial number of topics
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.gamma = gamma # parameter of tables prior
        self.docs = docs # a list of documents which include the words
        self.V = V # number of different words in the vocabulary
        self.z_m_n = {} # topic assignements for documents
        self.n_m_z = numpy.zeros((len(self.docs), self.K))      # number of words assigned to topic z in document m
        self.n_m = numpy.zeros(len(self.docs))     # number of words assigned to topic z in document m
#self.theta = numpy.zeros((len(self.docs), self.K))
        self.n_z_t = numpy.zeros((self.K, V)) # number of times a word v is assigned to a topic z
#self.phi = numpy.zeros((self.K, V))
        self.n_z = numpy.zeros(self.K)   # total number of words assigned to a topic z
        self.U1=[] # active topics
        for i in range (self.K):
            self.U1.append(i)
        
        self.U0=[] # deactive topics
        self.tau=numpy.zeros(self.K+1) +1./self.K
        for m, doc in enumerate(docs):         # Initialization of the data structures
            for n,t in enumerate(doc):
                z = numpy.random.randint(0, self.K) # Randomly assign a topic to a word and increase the counting array
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
                self.z_m_n[(m,n)]=z
    

    def inference(self,iteration):
        " Inference of HDP  using Dircet Assignment with ILDA simpilifying "
        self.iteration = iteration
        for m, doc in enumerate(self.docs):
                for n, t in enumerate(doc):
                    # decrease the counting for word t with topic kold
                    kold =self.z_m_n[(m,n)]
                    self.n_m_z[m,kold] -= 1
                    self.n_z_t[kold, t] -= 1
                    self.n_z[kold] -= 1
                    p_z=numpy.zeros(self.K+1)
                    for kk in range (self.K): # using the z sampling equation in ILDA
                        k=self.U1[kk]
                        p_z[kk]=(self.n_m_z[m,k]+self.alpha*self.tau[k])*(self.n_z_t[k,t]+self.beta)/(self.n_z[k]+self.V*self.beta)
                    p_z[self.K]=(self.alpha*self.tau[self.K])/self.V # additional cordinate for new topic
                    knew = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                    if knew==self.K: # check if topic sample is new
                        self.z_m_n[(m,n)] = self.spawntopic(m,t) # extend the number of topics and arrays shape and assign the array for new topic
                        self.updatetau() # update the table distribution over topic


                    else :
                        k=self.U1[knew] # do same as LDA
                        self.z_m_n[(m,n)] = k
                        self.n_m_z[m,k] += 1
                        self.n_z_t[k, t] += 1
                        self.n_z[k] += 1
                    
                    
                    if self.n_z[kold]==0: # check if the topic have been not used and re shape the arrayes
                        self.U1.remove(kold)
                        self.U0.append(kold)
                        self.K -=1
                        self.updatetau()
        for i in range(len(self.n_m_z)):
            self.n_m[i]=(numpy.sum(self.n_m_z[i]))
        print ('Iteration:',iteration,'\n','Number of topics:',self.K,'\n','Activated `topics:',self.U1,'\n','Deactivated topics',self.U0)

    def spawntopic (self,m,t): # reshape the arrays for new topic
        if len(self.U0)>0: # if the we have deactive topics.
            k=self.U0[0]
            self.U0.remove(k)
            self.U1.append(k)
            self.n_m_z[m,k]=1
            self.n_z_t[k,t]=1
            self.n_z[k]=1            
        else:
            k=self.K #  if the we do not have deactive topics so far.
            self.n_m_z=numpy.append(self.n_m_z,numpy.zeros([len(self.docs),1]),1)
            self.U1.append(k)
            self.n_m_z[m,k] = 1
            self.n_z_t=numpy.vstack([self.n_z_t,numpy.zeros(self.V)])
            self.n_z_t[k, t] = 1
            self.n_z=numpy.append(self.n_z,1)
            self.tau=numpy.append(self.tau,0)       
        self.K +=1
        return k    
            
    def stirling(self,nn): # making an array for keep the stirling(N,1:N) for saving time consumming
        if len(self.alss)==0:
            self.alss.append([])
            self.alss[0].append(1)
        if nn > self.maxnn:
            for mm in range (self.maxnn,nn):
                ln=len(self.alss[mm-1])+1
                self.alss.append([])               
                for xx in range(ln) :
                    #print(self.alss)
                    self.alss[mm].append(0)
                    if xx< (ln-1):
                        #if self.iteration >= 5 :
                        #    print(xx)
                        #    print('self.alss[mm-1][xx]=')
                        #    print(self.alss[mm-1][xx])
                        #    print('mm')
                        #    print(mm)
                        self.alss[mm][xx] += self.alss[mm-1][xx]*mm
                    if xx>(ln-2) :
                        self.alss[mm][xx] += 0
                    if xx==0 :
                        self.alss[mm][xx] += 0
                    if xx!=0 :
                        self.alss[mm][xx] += self.alss[mm-1][xx-1]

            self.maxnn=nn
        return self.alss[nn-1]
    
    
    
    def rand_antoniak(self,alpha, n):
        # Sample from Antoniak Distribution
        ss = self.stirling(n)
        max_val = max(ss)
        p = numpy.array(ss) / max_val
        
        aa = 1
        for i, _ in enumerate(p):
            p[i] *= aa
            aa *= alpha
        
        p = numpy.array(p,dtype='float') / numpy.array(p,dtype='float').sum()
        return choice(range(1, n+1), p=p)    
    
    
    def updatetau(self):  # update tau using antoniak sampling from CRM
    
        m_k=numpy.zeros(self.K+1)
        for kk in range(self.K):
            k=self.U1[kk]
            for m in range(len(self.docs)):
                
                if self.n_m_z[m,k]>1 :
                    m_k[kk]+=self.rand_antoniak(self.alpha*self.tau[k], int(self.n_m_z[m,k]))
                else :
                    m_k[kk]+=self.n_m_z[m,k]
    
        T=numpy.sum(m_k)
        m_k[self.K]=self.gamma
        tt=numpy.transpose(numpy.random.dirichlet(m_k, 1))
        for kk in range(self.K):
            k=self.U1[kk]
            self.tau[k]=tt[kk]

        self.tau[self.K]=tt[self.K]

    def _phi(self):
        """topic-word distribution, \phi in Blei'spaper  """
        self.phi = (self.n_z_t +self.beta)/ (self.n_z[:, numpy.newaxis]+self.V*self.beta)
        #return (self.n_z_t +self.beta)/ (self.n_z[:, numpy.newaxis]+self.V*self.beta),len(self.n_z)
        return self.phi,len(self.n_z)

    def _theta(self):
        """doc-topic distribution, \theta in Blei'spaper  """
        self.theta = (self.n_m_z +self.alpha)/ (self.n_m[:, numpy.newaxis]+self.K*self.alpha)
        #return (self.n_m_z +self.alpha)/ (self.n_m[:, numpy.newaxis]+self.K*self.alpha),len(self.n_m)
        return self.theta,len(self.n_m)

#量化指标计算
def f_perplexity(HDP,theta,phi,voca,newdoc):#计算内容困惑度
    K = len(phi)
    M = len(theta)
    V = len(phi[0])
    count = 0
    i = 0
    for m in range(M):
        l = newdoc[m]
        wordlist = [voca.vocas[id] for id in l]
        mul = 0
        for j in range(len(wordlist)):
            sum = 0
            word = wordlist[j]
            index = voca.vocas.index(word)
            for k in HDP.U1:
                sum = sum + phi[k][index]*theta[m][k]
            mul +=  math.log(sum)
        count += mul
    fenzi = 0 - count
    fenmu = f_word_count(newdoc)
    p = math.exp(fenzi/fenmu)
    return p
def f_word_count(testset):
    l = [len(line) for line in testset]
    count = 0
    for i in l:
        count += i
    return count

def KL(HDP,phi):#主题多样性（计算KL散度）-绘制热力图
    N = len(phi)
    K = len(HDP.U1)
    V = len(phi[0])
    D = numpy.zeros((K,K))
    for x,i in enumerate(HDP.U1):
        for y,j in enumerate(HDP.U1):
            if 1:
                s = 0
                for v in range(V):
                    t = phi[i][v]*math.log(phi[i][v]/phi[j][v])
                    s += t
                D[x][y] = s
    def trans(X):
        X = X.T
        return X
    def MaxMinNormalization(X):
        Max = numpy.max(X)
        Min = numpy.min(X)
        for i in range(len(X)):
            for j in range(len(X)):
                X[i][j] = (X[i][j] - Min) / (Max - Min)
        return X  
    D2 = trans(D)
    DD = (D+D2)/2
    Dx = MaxMinNormalization(DD)
    return Dx

def complexity(HDP,newdocs,phi):
    s = 0
    M = len(newdocs)
    K = len(HDP.U1)
    for k in HDP.U1:
        for j in range(M):
            for i in range(len(newdocs[j])):
                if HDP.z_m_n[(j,i)] == k:
                    s += 1
                    break
    c = K + s
    return c

#数据保存
def savewordidmap(voca):
    r1 = []
    r2 = []
    cfile = ".wordidmap.csv"
    for index,word in enumerate(voca.vocas):
        r1.append(index)
        r2.append(word)
    dataframe = pd.DataFrame({'index':r1,'word':r2})
    dataframe.to_csv(cfile,index=False,sep=',')
    print("词与序号对应关系已保存到wordidmap")

def savetheta(a):
    file = "./theta.csv"
    dataframe = pd.DataFrame(a)
    dataframe.to_csv(file,index=True,sep=',')
    print('文章-主题分布已保存到theta')
def savephi(a):
    file = "./phi.csv"
    dataframe = pd.DataFrame(a)
    dataframe.to_csv(file,index=True,sep=',')
    print('文章-主题分布已保存到phi')

def savetopn(HDP,phi,voca):
    zz = HDP.z_m_n
    len2 = len(HDP.n_z)
    count = [[]for t in range(len2)]
    for i in range(len2):
        for (d,t) in zz.items():
            if t == i:
                count[i].append(d[0])
        count[i] = len(list(set(count[i])))
    wordlist = [[]for i in range(len2)]
    prolist = [[]for i in range(len2)]
    for (i,U1) in enumerate(HDP.U1):
        x = phi[U1]
        dec = numpy.argsort(-x)
        ind = dec[0:15]#概率最大的15个词的索引
        for j in ind:
            pro = phi[U1][j]
            wordlist[i].append(voca[j])
            prolist[i].append(pro)
    cfile = "./topNfile.csv"
    with open (cfile,"w",newline='') as csvfile:
        writer = csv.writer(csvfile)
        fileheader = ["Active topics","Related documents","Key words","Probility"]
        writer.writerow(fileheader)
        l = []
        for i in range(len(count)):
            for j in range(len(wordlist[i])):
                t = [HDP.U1[i],count[i],wordlist[i][j],prolist[i][j]]
                l.append(t)
        writer.writerows(l)
    print("主题topN词已保存到到topNfile")

def saveper(index,per): #迭代数-困惑度
    cfile = "./per.csv"
    dataframe = pd.DataFrame({'iteration':index,'perplexity':per})
    dataframe.to_csv(cfile,index=False,sep=',')
def saveper2(tnums,per): #主题数-困惑度
    cfile = "./tper.csv"
    dataframe = pd.DataFrame({'topic number':tnums,'perplexity':per})
    dataframe.to_csv(cfile,index=False,sep=',')
def savecom(index,com): #迭代数-复杂度
    cfile = "./com.csv"
    dataframe = pd.DataFrame({'iteration':index,'complexity':com})
    dataframe.to_csv(cfile,index=False,sep=',')
def saveKL(a): #主题多样性
    cfile = "./KL.csv" 
    dataframe = pd.DataFrame(a)
    dataframe.to_csv(cfile,index=False,sep=',')
    print('KL距离已保存')
def saveit(a): #迭代数与主题数关系（主题独特性）
    r1 = []
    r2 = []
    cfile = "./iter-tops.csv"
    for index,num in enumerate(a):
        r1.append(index)
        r2.append(num)
    dataframe = pd.DataFrame({'iterations':r1,'topic numbers':r2})
    dataframe.to_csv(cfile,index=False,sep=',')
    print("迭代数与主题数关系已保存到iter-tops")

def hdp(alpha,beta,gamma):
    logging.basicConfig(filename="./4.logger/logger_hdp.log", format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    voca = prechinese.Vocabulary(excluds_stopwords=False) # find the unique words in the dataset
    corpus = codecs.open(segfile, 'r', encoding='utf8').read().splitlines() # toy data set to test the algorithm (1001 documents)
    print ("length of corpus:",len(corpus))
    docs = [voca.doc_to_ids(doc) for doc in corpus] # change words of the corpus to ids
    V1 = voca.size()
    print("初始词汇量：%d"%(V1))
    f1 = 0*len(docs)#词频筛选
    f2 = 1*len(docs)
    newdocs = docs
    newdocs = voca.choose_freq(docs,f1,f2)
    V2 = voca.size()
    print("处理后词汇量：%d"%(V2))
    HDP = HDP_gibbs_sampling(K0=K0, alpha=alpha, beta=beta, gamma=gamma, docs=newdocs, V=voca.size())
    savewordidmap(voca)
    return (HDP,voca,newdocs)

def newrun(HDP,voca,newdocs):#计算并保存最终结果
    logging.info("---------------迭代开始-------------" )
    iterations = max_iter #测试时修改,允许最大迭代次数
    per = []
    com = []
    index = []
    tnums = []
    for i in range(iterations):
        j = i + 1
        HDP.inference(i)
        logging.info("已完成迭代：%d" %(i))
        (theta,len1) = HDP._theta() # find word distribution of each topic
        (phi,len2) = HDP._phi() # find word distribution of each topic
        p = f_perplexity(HDP,theta,phi,voca,newdocs)
        c = complexity(HDP,newdocs,phi)
        n = len(HDP.U1)
        tnums.append(n)
        per.append(p)
        com.append(c)
        index.append(j)
        if i>min_iter and abs(per[j-1]-per[j-2]) < absper:#测试修改参数,absper=0.1,iter2=498
            savetheta(theta)
            savephi(phi)
            savetopn(HDP,phi,voca)
            saveper(index,per)
            saveper2(tnums,per)
            savecom(index,com)
            saveit(tnums)
            a = KL(HDP,phi)
            saveKL(a)
            break
    print("有效迭代次数为",str(j))
    return (index,per)


def argrun(HDP,voca,newdocs):
    logging.info("---------------迭代开始-------------" )
    iterations = arg_iter
    per = []
    com = []
    index = []
    tnums = []
    for i in range(iterations):
        j = i + 1
        HDP.inference(i)
        logging.info("已完成迭代：%d" %(i))
        (theta,len1) = HDP._theta() # find word distribution of each topic
        (phi,len2) = HDP._phi() # find word distribution of each topic
        p = f_perplexity(HDP,theta,phi,voca,newdocs)
        c = complexity(HDP,newdocs,phi)
        n = len(HDP.U1)
        tnums.append(n)
        per.append(p)
        com.append(c)
        index.append(j)
    #print("有效迭代次数为",str(j))
    return (index,per)

def setargs():#调参，得到困惑度最小的参数组合
    logging.basicConfig(filename="./logger_setargs.log", format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    voca = prechinese.Vocabulary(excluds_stopwords=False) # find the unique words in the dataset
    corpus = codecs.open(segfile, 'r', encoding='utf8').read().splitlines() # toy data set to test the algorithm (1001 documents)
    print ("length of corpus:",len(corpus))
    docs = [voca.doc_to_ids(doc) for doc in corpus] # change words of the corpus to ids
    V1 = voca.size()
    print("初始词汇量：%d"%(V1))
#newdocs = voca.choose_freq(docs,0,5000)
    newdocs = docs
    V2 = voca.size()
    print("处理后词汇量：%d"%(V2))
    
    alpha = [0.1,0.5,0.7,0.9,1]
    beta = [0.05,0.1,0.3,0.5]
    gamma = [0.125, 0.25, 0.5 ,1]
    perplexity = []
    args = dict()
    key_iteration = []
    key_alpha = []
    key_beta = []
    key_gamma = []
    for a in alpha:
        for b in beta:
            for g in gamma:
                HDP = HDP_gibbs_sampling(K0=K0, alpha=a, beta=b, gamma=g, docs=newdocs, V=voca.size())
                print("参数设置：{alpha=%s ,beta=%s,gamma=%s}" %(a,b,g))
                (index,per) = argrun(HDP,voca,newdocs)#执行argrun
                for index,pp in enumerate(per):
                    key_iteration.append(index)
                    key_alpha.append(a)
                    key_beta.append(b)
                    key_gamma.append(g)
                    perplexity.append(pp)
    args["iteration"] = key_iteration
    args["alpha"] = key_alpha
    args["beta"] = key_beta
    args["gamma"] = key_gamma
    args["perplexity"] = perplexity
    df = pd.DataFrame(args)
    index = df['perplexity'].idxmin()
    arg = dict()
    arg["iteration"] = [index]
    arg["alpha"] = [df["alpha"][index]]
    arg["beta"] = [df["beta"][index]]
    arg["gamma"] = [df["gamma"][index]]
    arg["perplexity"] = [df["perplexity"][index]]
    cfile = ".args.csv" 
    df.to_csv(cfile,index=False,sep=',')
    df2 = pd.DataFrame(arg)
    cfile2 = ".bestargs.csv" 
    df2.to_csv(cfile2,index=False,sep=',')
    print("最优参数设置：{alpha=%s ,beta=%s,gamma=%s,perplexity=%s}" %(arg["alpha"],arg["beta"],arg["gamma"],arg["perplexity"]))
    return (df["alpha"][index],df["beta"][index],df["gamma"][index])
                

if __name__ == "__main__":
    (alpha,beta,gamma) = setargs()
    HDP,voca,newdocs = hdp(alpha,beta,gamma)
    (index,per) = newrun(HDP,voca,newdocs)
