**Overview**

__In this branch, we perform validation using a _leave one subject out_ setting, in which a part of the subjects involved in
the experiment is used exclusively for training, and one subject exclusively for testing. The master branch uses a hold-out setting, with
70% training and 30% testing split__

This replication package includes all the data and scripts
required to replicate the experiments provided in the paper
"Using Voice and Biofeedback to Predict User Engagement during 
Requirements Interviews" by: Alessio Ferrari, Thaide Uichapa, Paola Spoletini, 
Nicole Novielli, Davide Fucci and Daniela Girardi.

The code in this project was developed by: Thaide Uichapa, Alessio Ferrari and Davide Fucci. 

For any issues, please contact: `alessio.ferrari AT isti.cnr.it`

**_Files_** are:

- main_multi_set_smote.py: this is the main script to perform the experiments.
- main_visualise_results.py: after running the experiments, a summary of the results
can be visualise with this script. 
- params.py: this includes all the dictionaries for the parameters used in parameter tuning.
- params_pipeline.py: this includes all the dictionaries for the parameters used in parameter tuning, 
with the naming required by scikit learn when a Pipeline object is used (in this case, to scale the data).
- Preprint.pdf: preprint of the paper associated to this package. 

**_Folders_** are:
- Data-imputation: this folder contains all the data for the different experiments, 
using imputation for the voice data. 
- Data-no-imputation: this folder contains all the data without imputation.
- Data-3-labels: this folder contains all the data with 3 labels for valence and arousal.
- Protocol: protocol description and associated files to replicate the experiment.

**Required Libraries**
All required libraries are listed in requirements.txt

If you are using Anaconda, then you can easily create a virtual environment
with all required packages as follows:

conda create --name <env> --file requirements.txt

**Usage**
To run all the experiments, with the different configurations, you need to:

1. Run the following command with the different combinations of options:

> python main_multi_set_smote.py <OVERSAMPLING> <SCALING> <SEARCH_TYPE> <IMPUTATION>

OVERSAMPLING: 'yes' or 'no', indicates whether you want to apply oversampling with SMOTE
SCALING: 'yes' or 'no', indicates whether data should be standardized
SEARCH_TYPE: 'grid' or 'random', indicates the type of search algorithm to use in hyperparameter tuning
IMPUTATION: 'yes' or 'no' or '3-labels'. Indicates whether the experiments should be run 
on data with imputation ('yes', folder 'Data-imputation' will be used), without imputation ('no', folder
'Data-no-imputation' is used, or 3-labels ('3-labels', folder 'Data-3-labels' is used)

The results will be stored in the <RESULT_FOLDER>, named:

> 'Results'+'-over-['+<OVERSAMPLING>+']-scale-['+<SCALING>+']-imp-['+<IMPUTATION>+']'

For example: 

> Results-over-[yes]-scale-[yes]-imp-[no]/

2. After each run, results are generated as CSV files in the Data folder, with the 
same name of the input files and the '-res' postfix. To visualise a summary of these files,
run the following command:

> python main_visualise_results.py <RESULT_FOLDER>

3. The process above need to be repeated each time the command at step 1 is run with
different options. 

WARNING: be aware that the option SCALING='no', when using SVM and MLP can lead to overly long computation times,
as these algorithms are designed to use scaled data. 



