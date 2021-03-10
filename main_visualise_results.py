import ast
import os

import pandas as pd

AROUSAL_BIOFEEDBACK = 'Data/ArousalBiofeedback-res.csv'
VALENCE_BIOFEEDBACK = 'Data/ValenceBiofeedback-res.csv'

AROUSAL_VOICE = 'Data/ArousalVoice-res.csv'
VALENCE_VOICE = 'Data/ValenceVoice-res.csv'

AROUSAL_COMPLETE = 'Data/ArousalCombineComplete-res.csv'
VALENCE_COMPLETE = 'Data/ValenceCombineComplete-res.csv'

file_list = [AROUSAL_BIOFEEDBACK,VALENCE_BIOFEEDBACK,AROUSAL_VOICE,VALENCE_VOICE,AROUSAL_COMPLETE,VALENCE_COMPLETE]

alg_names = ['SVM', 'MLP', 'DTree', 'NB', 'RNN']

AVG_TYPE = 'macro avg' #change to weighted avg to have results for weighted average

def main_visualise_results():

    for f_name in file_list:

        print('=====================================')
        print("Results for " + os.path.basename(f_name) + '\n')

        df = pd.read_csv(f_name, header=0)
        tdf = df.transpose()
        dict_results = tdf.to_dict()

        for k in dict_results.keys():
            print('==============' + alg_names[k] + '==============')

            dict_out_prec_rec_f1 = dict().fromkeys(dict_results[k].keys())
            dict_out_accuracy = dict().fromkeys(dict_results[k].keys())

            for iteration in dict_results[k].keys(): #iterates over the exectutions
                elem = dict_results[k][iteration]
                result = elem.partition('confusion')[0] #this is to take just the first part of the string, in which results are stored, and close it to make the dictionary well formed

                if result.startswith('{'):
                    full_table = ast.literal_eval(result[:-3] + '}') #close the dictionary, otherwise sintax error will occur, and then load it
                    dict_out_prec_rec_f1[iteration] = full_table['results'][AVG_TYPE] #output the macro average values
                    dict_out_accuracy[iteration] = full_table['results']['accuracy'] #output the accuracy values

            print("computing average over 10 runs")
            dict_out_prec_rec_f1.pop("Unnamed: 0") #eliminate the first column as not needed

            prec_val = [dict_out_prec_rec_f1[item]['precision'] for item in dict_out_prec_rec_f1.keys()]
            rec_val = [dict_out_prec_rec_f1[item]['recall'] for item in dict_out_prec_rec_f1.keys()]
            f1_val = [dict_out_prec_rec_f1[item]['f1-score'] for item in dict_out_prec_rec_f1.keys()]
            acc_val = [dict_out_accuracy[item] for item in dict_out_prec_rec_f1.keys()]

            print("average precision: " + str(sum(prec_val)/len(prec_val)))
            print("average recall: " + str(sum(rec_val) / len(rec_val)))
            print("average f1: " + str(sum(f1_val) / len(f1_val)))

            print("average accuracy: " + str(sum(acc_val) / len(acc_val)))

            print('\n')


main_visualise_results()
