# coding: utf-8
# 模型训练与对比分析

import time
import pandas as pd
from sklearn import metrics,cross_validation
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import xgboost
from xgboost.sklearn import XGBClassifier

#读取输入数据集，输入数据为经过数据融合与数据标准化后的数据
#基分类器
base_stock1 = pd.read_csv("./data201812/stockinfo/base_stock1.csv")
base_stock2 = pd.read_csv("./data201812/stockinfo/base_stock2.csv")
base_stock3 = pd.read_csv("./data201812/stockinfo/base_stock3.csv")
#技术分类
tech_stock1 = pd.read_csv("./data201812/stockinfo/tech_stock1.csv")
tech_stock2 = pd.read_csv("./data201812/stockinfo/tech_stock2.csv")
tech_stock3 = pd.read_csv("./data201812/stockinfo/tech_stock3.csv")
#主题分类，三类新闻，得到三种主题分类器
vdp1_stock1 = pd.read_csv("./data201812/stockinfo/vdp1_stock1.csv")
vdp1_stock2 = pd.read_csv("./data201812/stockinfo/vdp1_stock2.csv")
vdp1_stock3 = pd.read_csv("./data201812/stockinfo/vdp1_stock3.csv")
vdp2_stock1 = pd.read_csv("./data201812/stockinfo/vdp2_stock1.csv")
vdp2_stock2 = pd.read_csv("./data201812/stockinfo/vdp2_stock2.csv")
vdp2_stock3 = pd.read_csv("./data201812/stockinfo/vdp2_stock3.csv")
vdp3_stock1 = pd.read_csv("./data201812/stockinfo/vdp3_stock1.csv")
vdp3_stock2 = pd.read_csv("./data201812/stockinfo/vdp3_stock2.csv")
vdp3_stock3 = pd.read_csv("./data201812/stockinfo/vdp3_stock3.csv")

def processTrainAndTestData(data,term):
    newdata = data.copy()
    newdata.drop(newdata[['return1']],axis=1, inplace=True)
    X1 = newdata[:-193-term].values # 抽取80%训练集特征
    X2 = newdata[-194:].values # 抽取20%测试集特征
    y1 = data['return1'].apply(lambda x: 1 if x > 0 else 0)
    #y = r.loc[:, 'r1'] # 抽取训练集标签r1
    ya = y1[:-193-term].values # 抽取80%训练集标签
    yb = y1[-194:].values # 抽取20%训练集标签
    return X1, X2, ya, yb


def nbclf(X_train, y_train, X_test,y_test):
    start = time.time()
    print ("==GaussianNB==")
    nbModel = GaussianNB()
    y_pred = nbModel.fit(X_train, y_train).predict(X_test)
    stop = time.time()
    print ("time of GaussianNB:" + str(stop-start) + "秒")
    return y_pred

def svmclf(X_train, y_train, X_test,y_test):
    start = time.time()
    print ('==svm==')
    C_range = [1e-2, 1, 1e2]
    gamma_range = [1e-1, 1, 1e1]
    #kernel_range = ['linear','poly','rbf','sigmoid','precomputed']
    param_grid = dict(gamma=gamma_range, C=C_range)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid = GridSearchCV(svm.SVC(), param_grid, scoring="accuracy", n_jobs=-1, cv=kfold)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    stop = time.time()
    print ("time of SVM:" + str(stop-start) + "秒")
    return y_pred


def xgbclf(X_train, y_train, X_test, y_test):
    start = time.time()
    print ("==XGBoost==")
    one_to_left = st.beta(10, 1)  
    from_zero_positive = st.expon(0, 12)
    params = {  
        #常规参数
        #'booster': 'gbtree', #gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算 
        #'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
        #'nthread': -1,                  # cpu 线程数
        #'scale_pos_weight'#正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。
        #模型参数
        "n_estimators": [3,8,10,30, 60,80,100], #总共迭代的次数，即决策树的个数
         "max_depth": [6,8,10],         # 树的最大深度，典型值3-10.默认6
        #'early_stopping_rounds' #在验证集上，当连续n次迭代，分数没有提高后，提前终止训练。
        "colsample_bytree": [0.5, 0.05, 0.6,0.7,0.8,0.9],#训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。
        "subsample": one_to_left,#训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。
        #'min_child_weight' #默认值为1,值越大，越容易欠拟合；       
        #学习任务参数
        "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2], # eta 学习速率
        "gamma": [0.1,0.2, 0.5, 1],              # 惩罚项系数，节点分裂所需的最小损失函数下降值（值越大，算法越保守）
        'reg_alpha':[1e-5, 1e-2, 0.1,0.1, 0.2, 0.3, 1, 10],
        #'objective': 'multi:softmax',  # 多分类的问题
    }
    xgbreg = XGBClassifier(njobs=-1,
                          scale_pos_weight=1, 
                           objective='binary:logistic',
                           min_child_weight = 1
                          )
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    cls =RandomizedSearchCV(xgbreg, params, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = cls.fit(X_train, y_train,
                eval_set = [(X_test,y_test)],
                early_stopping_rounds = 20,
               )
    y_pred = cls.best_estimator_.predict(X_test)
    best_pa = cls.best_params_
    acc = metrics.accuracy_score(y_test, y_pred)
    stop = time.time()
    print ("time of XGBoost:" + str(stop-start) + "秒")
    return y_pred

def metrics_score(y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print ("accuracy_score : %.4g" %(accuracy))
    rec = metrics.recall_score(y_test,y_pred)
    print ("recall : %.4g" %(rec))
    f_score = metrics.f1_score(y_test, y_pred)
    print ("f1_score: %.4g" %(f_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print ("auc : %.4g" %(auc))
    pre = metrics.precision_score(y_test,y_pred)
    print ("pre : %.4g" %(pre))
    return accuracy, rec, auc, f_score, pre

def startToPredict(data=None):
    terms = [1]
#     start = time.time()
    predictFun = [nbclf, svmclf,xgbclf]
    results = [[] for i in range(6)]
    n = 1
    for index,term in enumerate(terms):
        X_train, X_test, y_train, y_test = processTrainAndTestData(data,term)
        accuracy_list = []
        rec_list = []
        auc_list = []
        f_score_list = []
        pre_list = []
        term_result = []
        for pf in predictFun:
            count = 0
            sum_acc = 0
            sum_rec = 0
            sum_auc = 0
            sum_f1 = 0
            sum_pre = 0
            for i in range(n):
                y_pred = pf(X_train, y_train, X_test,y_test)
                accuracy, rec, auc, f_score, pre = metrics_score(y_test, y_pred)
                if auc :
                    count +=1
                    sum_acc += accuracy
                    sum_rec += rec
                    sum_auc += auc
                    sum_f1 += f_score
                    sum_pre += pre
            if count > 0:       
                accuracy_list.append(sum_acc/count)
                rec_list.append(sum_rec/count)
                auc_list.append(sum_auc/count)
                pre_list.append(sum_pre/count)
                f_score_list.append(sum_f1/count)
        term_result.append(accuracy_list)
        term_result.append(rec_list)
        term_result.append(auc_list)
        term_result.append(f_score_list)
        term_result.append(pre_list)
        results[index].append(term_result)
        print ("=========================================================================")
    return results

datalist1 = [base_stock1,base_stock2,base_stock3]#基本分析法
datalist2 = [tech_stock1,tech_stock2,tech_stock3]#技术分析法
datalist3 = [vdp1_stock1,vdp1_stock2,vdp1_stock3]#主题1
datalist4 = [vdp2_stock1,vdp2_stock2,vdp2_stock3]#主题2
datalist5 = [vdp3_stock1,vdp3_stock2,vdp3_stock3]#主题3


for i,data in enumerate(datalist1):#分别在多个输入数据集上运行，得到实验结果
    print("--------------------------data"+str(i+1)+"--------------------------")
    results = startToPredict(data)
