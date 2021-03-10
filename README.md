**Overview**

This replication package includes all the data and scripts
required to replicate the experiments provided in the paper
"Using Voice and Biofeedback to Predict User Engagement during 
Requirements Interviews", currently submitted to 
the Empirical Software Engineering Journal. For any issues, please
contact: `alessio.ferrari AT isti.cnr.it`

**_Files_** are:

- main_multi_set_smote.py: this is the main script to perform the experiments.
- main_visualise_results.py: after running the experiments, a summary of the results
can be visualise with this script. 
- params.py: this includes all the dictionaries for the parameters used in parameter tuning.
- params_pipeline.py: this includes all the dictionaries for the parameters used in parameter tuning, 
with the naming required by scikit learn when a Pipeline object is used (in this case, to scale the data). 

**_Folders_** are:
- Data: this folder is empty. Before runnning the experiment, the files from one of the 
two other folders described below should be copy-pasted in this folder. The script
main_multi_set_smote.py uses this folder as data source, and also to produce results.
- Data-imputation: this folder contains all the data for the different experiments, 
using imputation for the voice data. 
- Data-no-imputation: this folder contains all the data without imputation.

**Required Libraries**
All required libraries are listed in requirements.txt

If you are using Anaconda, then you can easily create a virtual environment
with all required packages as follows:

conda create --name <env> --file requirements.txt

**Usage**
To run all the experiments, with the different configurations, you need to:

1. Copy in the Data folder all files from Data-imputation OR Data-no-imputation
 
2. Run the following command with the different combinations of options:

> python main_multi_set_smote.py <OVERSAMPLING> <SCALING> <SEARCH_TYPE>

OVERSAMPLING: 'yes' or 'no', indicates whether you want to apply oversampling with SMOTE
SCALING: 'yes' or 'no', indicates whether data should be standardized
SEARCH_TYPE: 'grid' or 'random', indicates the type of search algorithm to use in hyperparameter tuning

3. After each run, results are generated as CSV files in the Data folder, with the 
same name of the input files and the '-res' postfix. To visualise a summary of these files,
run the following command:

> python main_visualise_results.py

4. The process above need to be repeated each time the command at step 2 is run with
different options. Please notice that at each run results are overwritten in the Data folder.  



