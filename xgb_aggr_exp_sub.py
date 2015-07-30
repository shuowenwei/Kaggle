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
from sklearn.cross_validation import KFold, train_test_split
from pandasql import sqldf



def load_data():
    print ("Loading data ......")
    start = time.time()
    DV = feature_extraction.DictVectorizer(sparse=False)
    ENC = preprocessing.OneHotEncoder(sparse = False)
    POLY = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    pysqldf = lambda q: sqldf(q, globals())    

    catCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12'
                , 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']
    numCols = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14',  'T2_V1','T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14','T2_V15']

    groupCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12'
                , 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']    

    labelCol =['Hazard']
    idCol =['Id']

    #Load data
    trainData = pd.read_csv('train.csv')
    testData = pd.read_csv('test.csv')
    aggrData=trainData.append(testData)
    
    trainSize = trainData.shape[0]

    testSize = testData.shape[0]

    print ("Train: %d rows loaded" % (trainSize))
    print ("Test: %d rows loaded" % (testSize))

    #Label and id
    trainY=trainData[labelCol].values
    testId = testData[idCol].values
    
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
        aggr = sqldf("select %s, avg(Hazard) as avg_hazard_%s, count(1) as cnt_%s, avg(T2_V1) as avg_T2_V1_%s, avg(T2_V2) as avg_T2_V2_%s, avg(T1_V1) as avg_T1_V1_%s, avg(T1_V2) as avg_T1_V2_%s from aggrData group by 1" % (col, col,col, col, col, col ,col), locals())
        train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
        testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')

    #trainCatVecX = DV.fit_transform(trainCatX.T.to_dict().values())
    ENC.fit(np.vstack((trainCatTran,testCatTran)))
    #OneHotEncode labled catagorical columns
    trainCatX = ENC.transform(trainCatTran)
    testCatX = ENC.transform(testCatTran)       

    trainX = np.hstack((trainCatX,trainNum.values))
    testX = np.hstack((testCatX,testNum.values))

    print ("Loading finished in %0.3fs" % (time.time() - start))    

    # return trainX.toarray(), trainY, testX.toarray(), testY.values, testId
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


def xgboost_pred(train,labels,test, params):

    plst = list(params.items())

    offset = 3000
    #offset = int(train.shape[0]*0.25)
    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    #create a train and validation dmatrices 
    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    #train using early stopping and predict
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)
    print "Best score:", model.best_score
    print "Best iteration:", model.best_iteration
    preds1 = model.predict(xgtest,ntree_limit=model.best_iteration)


    #reverse train and labels and use different 5k for early stopping. 
    # this adds very little to the score but it is an option if you are concerned about using all the data. 
    train = train[::-1,:]
    labels = np.log(labels[::-1])

    xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)
    print "Best score:", model.best_score
    print "Best iteration:", model.best_iteration
    preds2 = model.predict(xgtest,ntree_limit=model.best_iteration)


    #combine predictions
    #since the metric only cares about relative rank we don't need to average
    preds = preds1*2.5 + preds2*7.5
    return preds

if __name__ == '__main__':
    #load train and test 
    # trainX, trainY, testX, testY, testId = load_data()
    start = time.time()

    params = {}
    params["objective"] = "reg:linear"
    params["eval_metric"] = "rmse"
    params["eta"] = 0.01
    params["min_child_weight"] = 5
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.8
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 8



    trainX, trainY, testX, testId = load_data()
    predY = xgboost_pred(trainX,trainY,testX, params)

    #generate solution
    preds = pd.DataFrame({"Id": testId.reshape(testId.shape[0]), "Hazard": predY})
    preds = preds.set_index('Id')
    preds.to_csv('xgboost_avg_hazard_cnt_t2v1v2.csv')

#cv 0.385563    