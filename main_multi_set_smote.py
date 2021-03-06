import os
import sys
from warnings import simplefilter
import pandas as pd
import random

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import params_pipeline
import params 

AROUSAL_BIOFEEDBACK = 'Data/ArousalBiofeedback.csv'
VALENCE_BIOFEEDBACK = 'Data/ValenceBiofeedback.csv'

AROUSAL_VOICE = 'Data/ArousalVoice.csv'
VALENCE_VOICE = 'Data/ValenceVoice.csv'

AROUSAL_COMPLETE = 'Data/ArousalCombine.csv'
VALENCE_COMPLETE = 'Data/ValenceCombine.csv'

file_list = [AROUSAL_BIOFEEDBACK,VALENCE_BIOFEEDBACK,AROUSAL_VOICE,VALENCE_VOICE,AROUSAL_COMPLETE,VALENCE_COMPLETE]

alg_names = ['SVM', 'MLP', 'DTree', 'NB', 'RNN']

tuned_mdl = dict.fromkeys(alg_names)

N_SPLITS = 3
CROSS_VAL = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random.SystemRandom().randint(1, 100)) #cross validation for hyperparameter tuning
SCORING = 'f1_macro' #scoring for hyperparameter tuning: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
ITER_NUM = 10 #number of repetitions

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def read_input(in_file):

    data = pd.read_csv(in_file)
    y = data['Label']
    X = data.drop(['folder #', 'filename', 'Label'], axis=1)
    return X, y

def run_algorithm(model, x_test, y_test):
    y_preds = model.predict(x_test)
    results = classification_report(y_test, y_preds, output_dict=True)
    print(results['macro avg'])
    print('accuracy: ' + str(results['accuracy']))

    print("confusion matrix")
    confusion_matrix(y_test, y_preds)
    xo = pd.crosstab(y_test, y_preds, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(xo)

    return results, xo


def tune_model(mdl, param_space, x_train, y_train, cv, scoring, tuning_type, scaling):
    print("using scaling: {}".format(scaling))
    print("hyper-params tuning: {}".format(tuning_type))

    # if scaling is enabled, create a pipeline rather than tuning the model directly
    if scaling == 'yes':
        mdl = make_pipeline(preprocessing.StandardScaler(), mdl)  
    if  tuning_type == 'grid':
        clf = GridSearchCV(mdl, param_space, scoring, n_jobs=-1, cv=cv)
        clf.fit(x_train, y_train)
    else:
        clf = RandomizedSearchCV(mdl, param_space, n_iter=500, scoring=scoring, n_jobs=-1, cv=cv, random_state=1)
        clf.fit(x_train, y_train)

    return clf


def tune_algorithm(alg_name,x_train, y_train, cv, scoring, tuning_type, scaling):
    mdl = None
    if scaling == 'yes':
        parameter_space = params_pipeline
    else:
        parameter_space = params

    if alg_name == 'SVM':
        mdl = SVC(verbose=False)
        parameter_space = parameter_space.parameter_space_SVM
    elif alg_name == 'MLP':
        mdl = MLPClassifier(max_iter=1000)
        parameter_space = parameter_space.parameter_space_MLP
    elif alg_name == 'DTree':
        mdl = DecisionTreeClassifier()
        parameter_space = parameter_space.parameter_space_DTree
    elif alg_name == 'NB':
        mdl = GaussianNB()
        parameter_space = parameter_space.parameter_space_NB
    elif alg_name == 'RNN':
        mdl = RandomForestClassifier()
        parameter_space = parameter_space.parameter_space_RNN
    else:
        print('Algorithm ' + alg_name + ' not found!')

    mdl = tune_model(mdl, parameter_space, x_train, y_train, cv, scoring, tuning_type, scaling)

    return mdl



def create_splits(iter_num, x, y, oversample):
    x_train_l, x_test_l, y_train_l, y_test_l = [], [], [], []

    for i in range(iter_num):
        rand_value = random.SystemRandom().randint(1, 100)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=rand_value, shuffle=True,
                                                            stratify=y)
        
        if oversample == 'yes':
            print("performing oversampling")
            oversampler = SMOTE(random_state=42)
            x_train, y_train = oversampler.fit_resample(x_train, y_train)

        x_train_l.append(x_train)
        x_test_l.append(x_test)
        y_train_l.append(y_train)
        y_test_l.append(y_test)

    return x_train_l, x_test_l, y_train_l, y_test_l



def init_results_dict():
    results_dict = dict.fromkeys(file_list)
    for file in results_dict.keys():
        results_dict[file] = dict.fromkeys(alg_names)
        for alg in results_dict[file].keys():
            results_dict[file][alg] = dict.fromkeys(map(str, range(ITER_NUM)))
            for item in results_dict[file][alg].keys():
                results_dict[file][alg][item] = dict({'results': None, 'confusion': None})
    return results_dict

def main_run(args):
    _, oversampling, scaling, params_search = args
    results_dict = init_results_dict()

    for in_file in file_list:
         print('RUNNING on file :' + in_file)
         x, y = read_input(in_file)

         x_train_l, x_test_l, y_train_l, y_test_l = create_splits(ITER_NUM, x, y, oversampling)

         for alg_name in alg_names:
             for i in range(ITER_NUM):
                 print('EXECUTION NUMBER: ', i)

                 print('Tuning ' + alg_name + '...')
                 tuned_mdl[alg_name] = tune_algorithm(alg_name, x_train_l[i], y_train_l[i], CROSS_VAL, SCORING, params_search, scaling)
                 print('Running ' + alg_name + ' on ' + in_file + '...')
                 results_dict[in_file][alg_name][str(i)]['results'], results_dict[in_file][alg_name][str(i)]['confusion'] = run_algorithm(tuned_mdl[alg_name], x_test_l[i], y_test_l[i])

    for in_file in file_list:
        pd.DataFrame.from_dict(results_dict[in_file], orient='index').to_csv(os.path.splitext(in_file)[0] + '-res.csv')

    print('DONE!')


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print("# 1 --> oversampling (yes) # 2 --> scaling (yes) # 3 --> search (grid) ")
        args = ['', 'yes', 'yes', 'grid']
    else:print(args)
    main_run(args)
