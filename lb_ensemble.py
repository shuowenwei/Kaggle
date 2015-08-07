
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import xgboost

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
    
    catCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']
    numCols = ['T1_V1', 'T1_V2', 'T1_V3', 'T1_V10', 'T1_V13', 'T1_V14',  'T2_V1','T2_V2', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'T2_V10', 'T2_V14','T2_V15']
    
    groupCols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5','T2_V11', 'T2_V12', 'T2_V13']
    
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


if __name__ == '__main__':
    
    np.random.seed(0) # seed to shuffle the train set
    
    n_folds = 10
    verbose = True
    shuffle = False
    
    X, y, X_submission, id_submission = load_data()
    y=y.reshape(50999)
    
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestRegressor(n_estimators=100),
        ExtraTreesRegressor(n_estimators=100),
        xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, min_child_weight=5, subsample=0.8, colsample_bytree =0.8, max_depth=8)]
    
    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test)
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(X_submission)
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LinearRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict(dataset_blend_test)

    preds = pd.DataFrame({"Id": id_submission.reshape(id_submission.shape[0]), "Hazard": y_submission})
    preds = preds.set_index('Id')
    preds.to_csv('lb_ensemble.csv')