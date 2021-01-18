from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split
import os
import pickle

def trainBasicClassifier(dataSet,lableColumnName,classDict_Reversed,classifierBasePath,shouldTrainModelRegardless):

    if not os.path.exists(classifierBasePath):
        os.makedirs(classifierBasePath)

    modelSavePath = os.path.join(classifierBasePath,'Classifier.pkl')
    triningData_Path = os.path.join(classifierBasePath,'TrainingData_X_Y_YNames.pkl')

    integarTypeColumns=[]
    for i,item in enumerate(dataSet.sum()):
        colName = dataSet.columns.values[i]
        if type(item)!=str:
            integarTypeColumns.append(colName)

    #integarTypeColumns = dataSet.describe().columns.values

    Y_Names = dataSet[lableColumnName].values
    Y = [classDict_Reversed[name.lower()] for name in Y_Names]
    X = dataSet.filter(items=integarTypeColumns, axis=1).copy()


    train_X, validation_X, train_y, validation_y = train_test_split(X, Y, test_size=0.2,
                                                                    random_state=101)
    traingSize = train_X.shape[0]

    if os.path.exists(modelSavePath) and os.path.exists(triningData_Path) and shouldTrainModelRegardless is False:
        classifierModel = pickle.load(open(modelSavePath, 'rb'))
        X,Y,Y_Names = pickle.load(open(triningData_Path, 'rb'))
        classifierModel.fit(X=train_X, y=train_y)
        score = classifierModel.score(validation_X, validation_y)

    else:

        classifierModel = RandomForestClassifier()
        classifierModel.fit(X=train_X, y=train_y)
        score = classifierModel.score(validation_X, validation_y)
        val_scores = cross_val_score(classifierModel, X, Y, cv=3)
        val_score = val_scores.mean()

        pickle.dump(classifierModel, open(modelSavePath, "wb"))
        pickle.dump((X,Y,Y_Names), open(triningData_Path, "wb"))


    headersForTraining = list(X.columns.values)

    print("Basic RainforestModel trained. Training size: %s Score: %s ,Val Score:%s"%(traingSize,round(score,3),round(val_score,3)))

    return classifierModel,headersForTraining,score,val_score

if __name__=='__main__':
    pass