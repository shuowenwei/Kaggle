import time
#import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble
from sklearn import feature_extraction
from sklearn import pipeline, metrics, grid_search
from sklearn.linear_model import ElasticNet
from sklearn.utils import shuffle
from scipy.sparse import hstack
from pandasql import sqldf
import xgboost as xgb


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
        aggr = sqldf("select %s, avg(Hazard) as avg_hazard_%s, count(1) as cnt_%s, avg(T2_V1) as avg_T2_V1_%s, avg(T2_V2) as avg_T2_V2_%s, avg(T1_V1) as avg_T1_V1_%s, avg(T1_V2) as avg_T1_V2_%s from aggrData group by 1" % (col, col,col, col, col, col ,col), locals())
        train_tmp = sqldf("select b.* from trainCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        test_tmp = sqldf("select b.* from testCat a left join aggr b on a.%s = b.%s " %(col,col),locals()).drop(col,axis=1)
        trainNum = pd.concat([trainNum, train_tmp],axis=1 ,join='inner')
        testNum = pd.concat([testNum, test_tmp],axis=1 ,join='inner')

    ENC.fit(np.vstack((trainCatTran,testCatTran)))
    #OneHotEncode labled catagorical columns
    trainCatX = ENC.transform(trainCatTran)
    testCatX = ENC.transform(testCatTran)

    trainX = np.hstack((trainCatX,trainNum.values))
    testX = np.hstack((testCatX,testNum.values))

    
    print ("Loading finished in %0.3fs" % (time.time() - start))
    
    return trainX, trainY, testX, testId

def Gini(y_true, y_pred):
	# check and get number of samples
	assert y_true.shape == y_pred.shape
	n_samples = y_true.shape[0]
	
	# sort rows on prediction column 
	# (from largest to smallest)
	arr = np.array([y_true, y_pred]).transpose()
	true_order = arr[arr[:,0].argsort()][::-1,0]
	pred_order = arr[arr[:,1].argsort()][::-1,0]
	
	# get Lorenz curves
	L_true = np.cumsum(true_order) / np.sum(true_order)
	L_pred = np.cumsum(pred_order) / np.sum(pred_order)
	L_ones = np.linspace(0, 1, n_samples)
	
	# get Gini coefficients (area between curves)
	G_true = np.sum(L_ones - L_true)
	G_pred = np.sum(L_ones - L_pred)
	
	# normalize to true Gini coefficient
	return G_pred/G_true


def search_model():
	trainX, trainY, _, _ = load_data()

	testX = trainX[40000:45000]
	testY = trainY[40000:45000]
	trainX = trainX[0:40000]
	trainY = trainY[0:40000]
    


	offset = trainX.shape[0]*0.3
	trainENX = trainX[0:offset]
	trainENY = trainY[0:offset]
        
	trainXGBX = trainX[offset:]
	trainXGBY = trainY[offset:]
	est = pipeline.Pipeline([('model', ElasticNet())])

	# Create a parameter grid to search for best parameters for everything in the pipeline
	param_grid = {'model__alpha': [0.0001,0.0003,0.001]
#					, 'model__normalize': [True,False]
					, 'model__l1_ratio' : [0.1,0.9]
				  }

	# Normalized Gini Scorer
	gini_scorer = metrics.make_scorer(Gini, greater_is_better = True)

	# Initialize Grid Search Model
	model = grid_search.GridSearchCV(estimator  = est,
									 param_grid = param_grid,
									 scoring	= gini_scorer,
									 #scoring	= "mean_squared_error",
									 verbose	= 10,
									 n_jobs	 = 1,
									 iid		= True,
									 refit	  = True,
									 cv		 = 4)
	# Fit Grid Search Model
	model.fit(trainENX, trainENY)
	print ("EN grid scores: ")
	print (model.grid_scores_)
	print("EN best score: %0.3f" % model.best_score_)
	print("EN best parameters set:", model.best_params_)
	print("EN best estimator:", model.best_estimator_)
	EN = model.best_estimator_
	EN.fit(trainENX, trainENY)
	trainENY = EN.predict(trainXGBX)
	predictENY = EN.predict(testX)
	print ("Val EN Gini score: ", Gini(testY,predictENY))
	trainXGBX = np.hstack([trainXGBX,trainENY.reshape(trainENY.shape[0],1)])
	testXGBX = np.hstack([testX,predictENY.reshape(predictENY.shape[0],1)])
    
        
	XGB = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, min_child_weight=5, subsample=0.8, colsample_bytree =0.8, max_depth=8)
    
	XGB.fit(trainX,trainY)
	pred = XGB.predict(testX)
	print ("Val XGB Gini score: ", Gini(testY,pred))
    
	XGB.fit(trainXGBX, trainXGBY)
	predictXGBY= XGB.predict(testXGBX)
    
	print ("Val EN-XGB Gini score: ", Gini(testY,predictXGBY))
    
	return model

def submit(model):
	# load test data
	_, _, testX, testId = load_data()
	testY = model.predict(testX)
	testDF = pd.DataFrame({"Id": testId, "Hazard": testY})
	testDF = testDF.set_index('Id')
	testDF.to_csv('en.csv')
	return


if __name__ == '__main__':

	start = time.time()
	print ("Starting...")
	model = search_model()
	#submit(model)
	print ("Finished in %0.3fs" % (time.time() - start))

# Degree 3
# Best score: 0.325
# Best parameters set:
#         model__alpha: 0.001
#         model__l1_ratio: 0.5
#         model__normalize: True
# Finished in 151.536s	


# Best score: 0.320
# ('Best parameters set:', {'model__normalize': True, 'model__alpha': 0.0007, 'model__l1_ratio
# ': 0.9})

#Best score: 0.322
#('Best parameters set:', {'model__normalize': True, 'model__alpha': 0.0009, 'model__l1_ratio': 1.0})
#Finished in 15.089s