import numpy as np

parameter_space_SVM = {
'C': [.0000001, .001, .1, 1, 10, 100, 1000],
'kernel': ['linear', 'poly', 'rbf']
}

parameter_space_MLP = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,100,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))

parameter_space_KNN = {
    'leaf_size': leaf_size,
    'n_neighbors': n_neighbors,
    'p': [1,2]
}

parameter_space_DTree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,4,6,8,10,12]
}

parameter_space_NB = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

parameter_space_RNN = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
