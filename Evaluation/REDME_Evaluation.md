# Evaluations

How to perform evaluations of the trained models.\
There are 2 ways for model evaluation:

1. Live evaluation using the experimental setup
2. Offline evaluation using the recorded data

## 1. Live evaluation

For live evaluation, [Live_Evaluate_Model.py](Live_Evaluate_Model.py) is used.\
Just enter your model path, model type and eventual preprocessing steps like normalizations and pca and run the script.

## 2. Offline evaluation

For offline evaluation, [Evaluate_Test_Set_Dataframe.py](Evaluate_Test_Set_Dataframe.py) is used.\
You need to collect data first using the Scripts in [Data_Generation](../Data_Generation).\
After collecting data you will have multiple dfs in a folder which you can combine using
[combine_datasets_and_convert_to_correct_format_for_training.py](../Data_Generation/combine_datasets_and_convert_to_correct_format_for_training.py).
Now you can add the path to the combined to [Evaluate_Test_Set_Dataframe.py](Evaluate_Test_Set_Dataframe.py) and run the
script.

## 3. Evaluation results

Both scripts will output a dataframe with the results of the evaluation.\
You can plot the results using [Plot_results_of_evaluation.py](Plot_results_of_evaluation.py).