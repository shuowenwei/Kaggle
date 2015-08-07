import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import time
from sklearn import preprocessing, ensemble
from sklearn import feature_extraction
from sklearn import pipeline, metrics, grid_search
from sklearn.utils import shuffle
from scipy.sparse import hstack
from sklearn.cross_validation import KFold, train_test_split,StratifiedKFold
from pandasql import sqldf
from scipy.stats import rankdata


def load_data():
    print ("Loading data ......")
    start = time.time()
    DV = feature_extraction.DictVectorizer(sparse=False)
    ENC = preprocessing.OneHotEncoder(sparse = False)
    POLY = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    pysqldf = lambda q: sqldf(q, globals())    

    catCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']
    numCols = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14',  'T2_V1','T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14','T2_V15']

    groupCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12' , 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']  

    T2NumCols = ['T2_V1','T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14','T2_V15']
    T2CatCols = ['T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']

    T1NumCols = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14']
    T1CatCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17',]

    labelCol =['Hazard']
    idCol =['Id']

    twoWayCatCols = [['T1_V17','T2_V12'],['T1_V5','T1_V9'],['T2_V13','T2_V11']] #T1_V5 T1_V9 , T1_V5 T1_V16, T2_V13 T2_V11

    #Load data
    trainData = pd.read_csv('train.csv')
    testData = pd.read_csv('test.csv')
    aggrData=trainData.append(testData)
    
    trainSize = trainData.shape[0]

    testSize = testData.shape[0]

    print ("Train: %d rows loaded" % (trainSize))
    print ("Test: %d rows loaded" % (testSize))

    #Label and id
    # trainY=trainData[labelCol].values
    trainY=trainData[labelCol].values.reshape(trainSize)
    testId = testData[idCol].values.reshape(testSize)
    
    #Categorical features
    trainCat=trainData[catCols]
    testCat=testData[catCols]

    #Numeric features
    trainNum=trainData[numCols]
    testNum=testData[numCols]

    #Label catagorical columns
    trainCatTran = pd.DataFrame()
    testCatTran = pd.DataFrame()
    
    #Encode categorical features
    for col in catCols:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(trainCat[col].values) + list(testCat[col].values))
        trainCatTran.insert(loc=0,column=col,value=lbl.transform(trainCat[col])) 
        testCatTran.insert(loc=0,column=col,value=lbl.transform(testCat[col])) 
        print (col)

    # for col in groupCols:
        # mean hazard
        aggr = aggrData.groupby(col)[labelCol].agg(np.mean) 
        aggr.rename(columns=lambda c: ('mean_'+c+'_BY_'+col).upper(), inplace = True) 
        train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
        testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')
        # mean numberic columns
        # aggr = aggrData.groupby(col)[numCols].agg(np.mean) 
        # aggr.rename(columns=lambda c: ('mean_'+c+'_BY_'+col).upper(), inplace = True) 
        # train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        # test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        # trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
        # testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')        
        # # median hazard
        # aggr = aggrData.groupby(col)[labelCol].agg(np.median) 
        # aggr.rename(columns=lambda c: ('median_'+c+'_BY_'+col).upper(), inplace = True) 
        # train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        # test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        # trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
        # testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')
        # count
        aggr = aggrData.groupby(col)[labelCol].agg(np.size) 
        aggr.rename(columns=lambda c: ('count_'+c+'_BY_'+col).upper(), inplace = True) 
        train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
        testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')
        print (col)


#two way interactions
    # for col in twoWayCatCols:
    #     aggr = aggrData.groupby(col)[labelCol].agg(np.mean) 
    #     aggr.rename(columns=lambda c: ('mean_'+c+'_BY_'+col[0]+'_'+col[1]).upper(), inplace = True) 
    #     train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s and a.%s = b.%s" %(col[0],col[0],col[1],col[1]),locals()).drop(col,axis=1)
    #     test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s and a.%s = b.%s" %(col[0],col[0],col[1],col[1]),locals()).drop(col,axis=1)
    #     trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
    #     testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')        

# #T2 by T1
#     for col in T1CatCols:
#         aggr = aggrData.groupby(col)[T2NumCols].agg(np.mean) 
#         aggr.rename(columns=lambda c: ('mean_'+c+'_BY_'+col).upper(), inplace = True) 
#         train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
#         test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
#         trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
#         testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')

# #T1 by T2
#     for col in T2CatCols:
#         aggr = aggrData.groupby(col)[T1NumCols].agg(np.mean) 
#         aggr.rename(columns=lambda c: ('mean_'+c+'_BY_'+col).upper(), inplace = True) 
#         train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
#         test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
#         trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
#         testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')        


    #trainCatVecX = DV.fit_transform(trainCatX.T.to_dict().values())
    ENC.fit(np.vstack((trainCatTran,testCatTran)))
    #OneHotEncode labled catagorical columns
    trainCatX = ENC.transform(trainCatTran)
    testCatX = ENC.transform(testCatTran)       

    trainX = np.hstack((trainCatX,trainNum.values))
    testX = np.hstack((testCatX,testNum.values))

    print ("Loading finished in %0.3fs" % (time.time() - start))    
    return trainX, trainY, testX, testId



def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini  


def xgboost_pred(train,labels,test,test_labels, params):

    plst = list(params.items())

    # offset = 4000
    offset = int(train.shape[0]*0.25)
    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices
    xtrain = xgb.DMatrix(train, labels)
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)
    print "Best score:", model.best_score
    print "Best iteration:", model.best_iteration
    model = xgb.train(plst, xtrain, model.best_iteration)
    preds1 = model.predict(xgtest)

    print ("score1: %f" % (normalized_gini(test_labels,preds1)))


    #reverse train and labels and use different 5k for early stopping. 
    # this adds very little to the score but it is an option if you are concerned about using all the data. 
    train = train[::-1,:]
    labels = np.log(labels[::-1])
    xtrain = xgb.DMatrix(train, labels)

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)
    print "Best score:", model.best_score
    print "Best iteration:", model.best_iteration
    model = xgb.train(plst, xtrain, model.best_iteration)
    preds2 = model.predict(xgtest)
    print ("score2: %f" % (normalized_gini(test_labels,preds2)))
    

    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    preds = preds1*2.5 + preds2*7.5
    return preds,preds1,preds2

if __name__ == '__main__':
    #load train and test 
    # trainX, trainY, testX, testY, testId = load_data()
    trainX, trainY, testX, testId = load_data()
    #Using 10000 samples
    # idx = np.random.choice(trainX.shape[0],10000)
    
#    trainX=trainX[30000:40000]
#    trainY=trainY[30000:40000]

    start = time.time()
    params = {}
    params["objective"] = "reg:linear"
    params["eval_metric"] = "rmse"
    params["eta"] = 0.01
    params["min_child_weight"] = 100
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.6
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 8    

    skf = StratifiedKFold(trainY, n_folds=4, random_state  = 42)
    # kf = KFold(trainX.shape[0], n_folds=4,shuffle=True, random_state  = 42 )
    gini_scores = []
    for train_index, test_index in skf:
        #Splict train set into k folds
        X_train_fold, X_test_fold = trainX[train_index], trainX[test_index]
        y_train_fold, y_test_fold = trainY[train_index], trainY[test_index]    
        y_pred_fold,y_pred_fold1,y_pred_fold2 = xgboost_pred(X_train_fold,y_train_fold,X_test_fold,y_test_fold, params)
        gini_scores.append([normalized_gini(y_test_fold,y_pred_fold1)
                            , normalized_gini(y_test_fold,y_pred_fold2)
                            , normalized_gini(y_test_fold,y_pred_fold)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')+rankdata(y_pred_fold2,method='dense'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')*1.5+rankdata(y_pred_fold2,method='dense'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')*2+rankdata(y_pred_fold2,method='dense'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')*3+rankdata(y_pred_fold2,method='dense'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')*4+rankdata(y_pred_fold2,method='dense'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')*5+rankdata(y_pred_fold2,method='dense'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')+rankdata(y_pred_fold2,method='dense')*1.5)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')+rankdata(y_pred_fold2,method='dense')*2)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')+rankdata(y_pred_fold2,method='dense')*2)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')+rankdata(y_pred_fold2,method='dense')*2)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='dense')+rankdata(y_pred_fold2,method='dense')*2)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')+rankdata(y_pred_fold2,method='average'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')*1.5+rankdata(y_pred_fold2,method='average'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')*2+rankdata(y_pred_fold2,method='average'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')*3+rankdata(y_pred_fold2,method='average'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')*4+rankdata(y_pred_fold2,method='average'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')*5+rankdata(y_pred_fold2,method='average'))
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')+rankdata(y_pred_fold2,method='average')*1.5)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')+rankdata(y_pred_fold2,method='average')*2)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')+rankdata(y_pred_fold2,method='average')*2)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')+rankdata(y_pred_fold2,method='average')*2)
                            , normalized_gini(y_test_fold,rankdata(y_pred_fold1,method='average')+rankdata(y_pred_fold2,method='average')*2)                            
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*2 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*3 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*4 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*5 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*5.5 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*6 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*7 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*8 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*9 )
                            , normalized_gini(y_test_fold,y_pred_fold1 +y_pred_fold2*10 )])
    
    gini_scores=np.array(gini_scores)   
    for i in gini_scores.T:
        print np.mean(i)
    print ("Finished in %0.3fs" % (time.time() - start))    

# val 1: 
    # catCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']
    # numCols = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14',  'T2_V1','T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14','T2_V15']

    # params["objective"] = "reg:linear"
    # params["eval_metric"] = "rmse"
    # params["eta"] = 0.01
    # params["min_child_weight"] = 5
    # params["subsample"] = 0.8
    # params["colsample_bytree"] = 0.8
    # params["scale_pos_weight"] = 1.0
    # params["silent"] = 1
    # params["max_depth"] = 8
# val 2: 0.359665 w encoded cat cols
# val 3: 0.365631, CV full 0.386412,  w encoded cat cols and mean hazard
# val 3.0.0: 0.366080 min_child_weight 10 w encoded cat cols and mean hazard
# val 3.0.1: 0.36696 min_child_weight 20 w encoded cat cols and mean hazard
# val 3.0.2: 0.369441 min_child_weight 50 w encoded cat cols and mean hazard
# val 3.0.3: 0.370005, CV full 0.389014, min_child_weight 100 w encoded cat cols and mean hazard
# val 3.0.3.3: 0.370301800205 min_child_weight 100, colsample_bytree 0.7, subsample 0.7,  encoded cat cols and mean hazard
# val 3.0.3.4: 0.370771084286, CV full 0.388882359727, p1+p2*6 0.389227382613 min_child_weight 100, colsample_bytree 0.6, subsample 0.7,  encoded cat cols and mean hazard and count
# val 3.0.3.4.1: 0.37020554459 min_child_weight 100, colsample_bytree 0.6, subsample 0.7,  encoded cat cols and mean hazard and count and mean num cols
# val 3.0.3.4.1: 0.369193684279 min_child_weight 100, colsample_bytree 0.7, subsample 0.7, max_depth 10 encoded cat cols and mean hazard and count and mean num cols
# val 3.0.3.4.1: 0.369086760753 min_child_weight 100, colsample_bytree 0.5, subsample 0.7,  encoded cat cols and mean hazard and count and mean num cols

# val 3.0.3.4: 0.370069007244 min_child_weight 100, colsample_bytree 0.6, subsample 0.7,  encoded cat cols and mean num cols
# val 3.0.3.4: 0.36938448236 min_child_weight 100, colsample_bytree 0.7, subsample 0.7,  encoded cat cols and mean num cols
# val 3.0.3.4: 0.368985378876 min_child_weight 100, colsample_bytree 0.5, subsample 0.7,  encoded cat cols and mean num cols
# val 3.0.3.4: 0.368624025695 min_child_weight 100, colsample_bytree 1.0, subsample 0.7,  encoded cat cols and mean num cols

# val 3.0.3.4: 0.36563760751 min_child_weight 5, colsample_bytree 0.8, subsample 0.8,  encoded cat cols and mean hazard by two way cols
# val 3.0.3.4: 0.361726274627 min_child_weight 100, colsample_bytree 0.8, subsample 0.8,  encoded cat cols and mean hazard by two way cols


# val 3.0.3.1: 0.352523 min_child_weight 200 w encoded cat cols and mean hazard
# val 3.0.3.2: 0.371108, CV full 0.388720, min_child_weight 80 w encoded cat cols and mean hazard!!!!!!!!!!!!!!!!!!!!!!!!!
# val 3.0.4: 0.340508, CV full 0.386874, LB 0.385404 , min_child_weight 300 w encoded cat cols and mean hazard
# val 3.0: 0.364200 colsample_bytree 1.0 w encoded cat cols and mean hazard
# val 3.1: 0.359431 w encoded cat cols and count
# val 3.2:  0.363566 w encoded cat cols and median hazard
# val 4: 0.363189 w encoded cat cols and mean hazard and count
# val 4.1: 0.363368 colsample_bytree 1.0 w encoded cat cols and mean hazard and count
# val 4.2: 0.365437 colsample_bytree 0.4 w encoded cat cols and mean hazard and count
# val 4.3: 0.362638 colsample_bytree 0.2 w encoded cat cols and mean hazard and count
# val 4.4: 0.365122 colsample_bytree 0.5 w encoded cat cols and mean+median hazard and count
# val 4.5: 0.361819 colsample_bytree 0.8 w encoded cat cols and mean+median hazard and count
# val 4.6: 0.364775 colsample_bytree 0.3 w encoded cat cols and mean+median hazard and count
# val 5: 0.363542 T1_V10 as num col, w encoded cat cols, 
# val 6: 0.364262 T1_V10 as num col, w encoded cat cols and mean hazard
# val 7: 0.362864 T1_V10 as num col, w encoded cat cols and mean hazard and count
# val 10: 0.360562 w encoded cat cols and mean numCols
# val 10.1: 0.362021 colsample_bytree 0.5 w encoded cat cols and mean numCols
# val 10.2: 0.356022 colsample_bytree 0.25 w encoded cat cols and mean numCols
# val 10.2: 0.360736 colsample_bytree 1 w encoded cat cols and mean numCols













#Mean gini score: 0.355645 with original tuned aggr


#Mean gini score: 0.358885 with encoded categorical columns - all

#Mean gini score: 0.361525 with encoded categorical columns - all, average harzad for all categorical columns
# 0.385563 cv 0.384648 lb

#Mean gini score: 0.365254 with encoded categorical columns - all, average harzad for all categorical columns, params from Kaggle scripts sqldf groupby
#0.387473 lb
#0.379417 lb with max_depth=15

#Mean gini score: 0.367663 with encoded categorical columns - all, average harzad for all categorical columns, params from Kaggle scripts - pd groupby

#Mean gini score: 0.366584 with encoded categorical columns - all, average harzad for all categorical columns,
# params
#     params["eta"] = 0.005
#     params["min_child_weight"] = 6
#     params["subsample"] = 0.7
#     params["colsample_bytree"] = 0.7
#     params["scale_pos_weight"] = 1.0
#     params["silent"] = 1
#     params["max_depth"] = 9 preds = preds1*1.6 + preds2*8

# Mean gini score: 0.364542 with encoded categorical columns - all, average harzad and numeric cols for all categorical columns, params from Kaggle scripts - pd groupby

#Mean gini score: 0.366884 with encoded categorical columns - all, median harzad for all categorical columns, params from Kaggle scripts - pd groupby

#Mean gini score: 0.363146 with encoded categorical columns - all,count by cat cols(all) ,params from Kaggle scripts pd groupby

# Mean gini score: 0.366601 with encoded categorical columns - all, average harzad by all categorical columns, T2_V1, T2_V2 BY T2_V3, T2_V5 from Kaggle scripts - pd groupby
# Mean gini score: 0.366022 with encoded categorical columns - all, average harzad by all categorical columns, T2 num cols BY T2 cat cols from Kaggle scripts - pd groupby
# Mean gini score: 0.366538 with encoded categorical columns - all, average harzad by all categorical columns, T1 num cols BY T1 cat cols from Kaggle scripts - pd groupby
# Mean gini score: 0.366745 with encoded categorical columns - all, average harzad by all categorical columns, T1 num cols BY T1 cat cols and T2 num cols BY T2 cat cols from Kaggle scripts - pd groupby
# Mean gini score: 0.367662 with encoded categorical columns - all, average harzad by all categorical columns, T1 num cols BY T2 cat cols and T2 num cols BY T1 cat cols from Kaggle scripts - pd groupby interesting!

#Mean gini score: 0.364471 with encoded categorical columns - all, average harzad for all categorical columns, count by cat cols(train only) ,params from Kaggle scripts

#Mean gini score: 0.365360 with encoded categorical columns - all, average harzad for all categorical columns, count by cat cols(all) ,params from Kaggle scripts


#Mean gini score: 0.366737 with encoded categorical columns - all, average harzad for all categorical columns, count, avg t2_v1, avg_t2_v2 by cat cols(all) ,params from Kaggle scripts
#0.386180 lb
