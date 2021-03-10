import numpy as np

parameter_space_SVM = {
'svc__C': [.0000001, .001, .1, 1, 10, 100, 1000],
'svc__kernel': ['linear', 'poly', 'rbf']
}

parameter_space_MLP = {
    'mlpclassifier__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,100,100)],
    'mlpclassifier__activation': ['tanh', 'relu'],
    'mlpclassifier__solver': ['sgd', 'adam'],
    'mlpclassifier__alpha': [0.0001, 0.05],
    'mlpclassifier__learning_rate': ['constant','adaptive'],
}

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))

parameter_space_KNN = {
    'kneighborsclassifier__leaf_size': leaf_size,
    'kneighborsclassifier__n_neighbors': n_neighbors,
    'kneighborsclassifier__p': [1,2]
}

parameter_space_DTree = {
    'decisiontreeclassifier__criterion': ['gini', 'entropy'],
    'decisiontreeclassifier__max_depth': [2,4,6,8,10,12]
}

parameter_space_NB = {
    'gaussiannb__var_smoothing': np.logspace(0,-9, num=100)
}

parameter_space_RNN = {
    'randomforestclassifier__bootstrap': [True],
    'randomforestclassifier__max_depth': [80, 90, 100, 110],
    'randomforestclassifier__max_features': [2, 3],
    'randomforestclassifier__min_samples_leaf': [3, 4, 5],
    'randomforestclassifier__min_samples_split': [8, 10, 12],
    'randomforestclassifier__n_estimators': [100, 200, 300, 1000]
}
