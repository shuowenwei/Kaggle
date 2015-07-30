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


def load_data():
    print ("Loading data ......")
    start = time.time()
    DV = feature_extraction.DictVectorizer(sparse=False)
    ENC = preprocessing.OneHotEncoder()
    POLY = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    catCols = ['T1_V3','T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12'
                , 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V6', 'T2_V11', 'T2_V12', 'T2_V13']
    # numCols = ['T1_V1','T1_V2', 'T1_V10', 'T1_V13' ,'T1_V14', 'T2_V1','T2_V2', 'T2_V4','T2_V7', 'T2_V9','T2_V10','T2_V14','T2_V15']
    numCols = ['T2_V1','T2_V2', 'T2_V4','T2_V7','T2_V9','T2_V10','T2_V15']

    groupCols = ['T1_V3','T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12'
                , 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V6', 'T2_V11', 'T2_V12', 'T2_V13']

    labelCol =['Hazard']
    idCol =['Id']

    #Load data
    trainData = pd.read_csv('C:\\Users\\Public\\Documents\\cyy\\ML\\Liberty\\Data\\train.csv')
    testData = pd.read_csv('C:\\Users\\Public\\Documents\\cyy\\ML\\Liberty\\Data\\test.csv')
    trainSize = trainData.shape[0]

    testSize = testData.shape[0]

    print ("Train: %d rows loaded" % (trainSize))
    print ("Test: %d rows loaded" % (testSize))

    #Label and id
    trainY=trainData['Hazard'].values
    testId = testData['Id'].values
    
    #Categorical features
    trainCat=trainData[catCols]
    testCat=testData[catCols]

    #Numeric features
    trainNumX=trainData.drop(catCols + labelCol + idCol, axis=1)
    testNumX=testData.drop(catCols + idCol, axis=1)

    #Aggregated features: median
    trainAggr = trainData[idCol + catCols + numCols]
    testAggr = testData[idCol + catCols + numCols]

    # log of numeric features - not work
    # trainAggr = trainData[idCol + catCols]
    # trainAggr[numCols] = np.log(trainData[numCols])
    # testAggr = testData[idCol + catCols]
    # testAggr[numCols] = np.log(testData[numCols])

    aggrData = pd.concat([trainAggr, testAggr])
    

    # Interacte numeric features
    # trainNumX=POLY.fit_transform(trainNumX)
    # testNumX=POLY.transform(testNumX)
    trainPolyX = POLY.fit_transform(trainData[['T1_V2','T2_V2']])
    testPolyX = POLY.fit_transform(testData[['T1_V2','T2_V2']])

    # trainPolyX = np.hstack([trainPolyX, POLY.fit_transform(trainData[['T2_V14','T2_V6']])])
    # testPolyX = np.hstack([testPolyX, POLY.fit_transform(testData[['T2_V14','T2_V6']])])

    #Label catagorical columns
    trainCatTran = pd.DataFrame()
    testCatTran = pd.DataFrame()

    #Encode categorical features
    for col in catCols:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(trainCat[col].values) + list(testCat[col].values))
        trainCatTran.insert(loc=0,column=col,value=lbl.transform(trainCat[col])) 
        testCatTran.insert(loc=0,column=col,value=lbl.transform(testCat[col])) 

    #trainCatVecX = DV.fit_transform(trainCatX.T.to_dict().values())
    ENC.fit(np.vstack((trainCatTran,testCatTran)))
    #OneHotEncode labled catagorical columns
    trainCatX = ENC.transform(trainCatTran)
    testCatX = ENC.transform(testCatTran)       

    #Aggregate numeric colu`mns - median

    for col in groupCols:
        aggr = aggrData.groupby(col)[numCols].transform(np.median)
        trainNumX = hstack([trainNumX,aggr[:trainSize]])
        testNumX = hstack([testNumX,aggr[trainSize:]])

        # aggr = aggrData.groupby(col)[numCols].agg([np.median])
        # aggr[col] = aggr.index
        # trainAggr = pd.merge(trainAggr, aggr, on=col, how='left')
        # testAggr = pd.merge(testAggr, aggr, on=col, how='left')
    #Remove catagorical/numerical/id from aggregated data
    # trainAggrX = trainAggr.drop(idCol + catCols + numCols, axis=1)
    # testAggrX = testAggr.drop(idCol + catCols + numCols, axis=1)

    # trainX = hstack((trainCatX,trainNumX, trainAggrX))
    # testX = hstack((testCatX,testNumX, testAggrX))

    trainX = hstack((trainCatX,trainNumX,trainPolyX))
    testX = hstack((testCatX,testNumX,testPolyX))

    print ("Loading finished in %0.3fs" % (time.time() - start))    

    # return trainX.toarray(), trainY, testX.toarray(), testY.values, testId
    return trainX.toarray(), trainY, testX.toarray(), testId



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

    #offset = 3000
    offset = int(train.shape[0]*0.25)
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

    params1 = {}
    params1["objective"] = "reg:linear"
    params1["eta"] = 0.01
    params1["min_child_weight"] = 5
    params1["subsample"] = 0.8
    params1["colsample_bytree"] = 0.8
    params1["scale_pos_weight"] = 1.0
    params1["silent"] = 1
    params1["max_depth"] = 8    

    params2 = {}
    params2["objective"] = "reg:linear"
    params2["eta"] = 0.01
    params2["min_child_weight"] = 6
    params2["subsample"] = 0.8
    params2["colsample_bytree"] = 0.8
    params2["scale_pos_weight"] = 1.0
    params2["silent"] = 1
    params2["max_depth"] = 10



    trainX, trainY, testX, testId = load_data()
    #Using 10000 samples
    # idx = np.random.choice(trainX.shape[0],10000)
    trainX=trainX[30000:40000]
    trainY=trainY[30000:40000]

    kf = KFold(trainX.shape[0], n_folds=4,shuffle=True, random_state  = 42 )
    gini_scores = []
    gini_scores1 = []
    gini_scores2 = []
    for train_index, test_index in kf:
        #Splict train set into k folds
        X_train_fold, X_test_fold = trainX[train_index], trainX[test_index]
        y_train_fold, y_test_fold = trainY[train_index], trainY[test_index]    

        y_pred_fold1 = xgboost_pred(X_train_fold,y_train_fold,X_test_fold, params1)
        y_pred_fold2 = xgboost_pred(X_train_fold,y_train_fold,X_test_fold, params2)

        gini_score1 = normalized_gini(y_test_fold,y_pred_fold1)
        gini_score2 = normalized_gini(y_test_fold,y_pred_fold2)
        gini_score = [normalized_gini(y_test_fold,y_pred_fold1*p + y_pred_fold2*(1-p)) for p in  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
        print gini_score1, gini_score2, gini_score
        gini_scores1.append(gini_score1)
        gini_scores2.append(gini_score2)
        gini_scores.append(gini_score)

    
    print ("Mean gini score: %f" % (np.mean(gini_scores)))
    print ("Loading finished in %0.3fs" % (time.time() - start))    
#   Mean gini score: 0.386984 
#   [0.391596323343095, 0.3856458758991576, 0.38958890237127125, 0.381105993125543]

# Mean gini score: 0.385977 without aggr
# [0.3889893278915356, 0.3838288271059869, 0.39007985449752053, 0.38101111772943796]

# Mean gini score: 0.386003 with T1 numerics


# In [33]: gini_scores1 Out[36]: 0.38497457598945545
# Out[33]:
# [0.39056555009847793,
#  0.3842796743326927,
#  0.38788428037676576,
#  0.37716879914988527]

# In [34]: gini_scores2 Out[37]: 0.38264380730040914
# Out[34]:
# [0.38643971869274174,
#  0.3829593498526453,
#  0.3858968177588837,
#  0.3752793428973657]

# In [35]: gini_scores
# Out[35]:
# [[0.3871617549140542,
#   0.3878196445888668,
#   0.38840666554541325,
#   0.3889402141034586,
#   0.38940802770012445,
#   0.38977540900078733,
#   0.3900903027201722,
#   0.390347644991639,
#   0.3904864851594307],
#  [0.383340600700439,
#   0.3836957319356017,
#   0.3839852544204646,
#   0.38420807123453354,
#   0.38438564330458835,
#   0.3844583497757138,
#   0.3845151057324064,
#   0.38448303693923436,
#   0.3844128340278173],
#  [0.38638013417148814,
#   0.38682780661316685,
#   0.3871647229008455,
#   0.3874499512808256,
#   0.387666892375639,
#   0.3878520903532178,
#   0.38796112911008607,
#   0.38798240823512914,
#   0.3879721676961043],
#  [0.3758060326873538,
#   0.3762314742144472,
#   0.37659622208553417,
#   0.3768471142144661,
#   0.3770396695947414,
#   0.3772028492191777,
#   0.37728545289657467,
#   0.3773147825676344,
#   0.3772660907051943]]

# In [36]: gini_scores