## Predicting Diabetes: A Comparative Study of Machine Learning Algorithms for Classification
This project is a part of the AAI-501-IN2 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

**Project Status: [Completed]**

### Installation
To Clone project from GitHub:
1. Github url for code repo: https://github.com/ramindersinghusd/aai-501-in2-project-group7
2. To run locally, clone the repo as below
```
git init
git clone https://github.com/ramindersinghusd/aai-501-in2-project-group7.git
```

Below local env setup used for development.

1. Install python: 3.11.3
2. Installed VSCode
3. Add Python and jupyter extension to VSCode
4. Set kernel in VSCode to execute notebook
5. conda install -n base ipykernel jupyter
6. conda -V >> conda 23.5.2
7. pip install jupyter notebook pandas numpy matplotlib scipy scikit-learn pandoc
nbconvert[webpdf] nbconvert notebook-as-pdf seaborn xgboost shap openpyxl
8. goto project folder and run >> jupyter notebook
9. This will open default browser at http://localhost:8080/tree 
10. Select the required notebook file and double click to run
11. This will open in new tab, the run all cell or execute specific cell

### Project Objective

The main purpose of this project is to use machine learning to predict diabetes with the help of a dataset consisting of several health characteristics. Such models as logistic regression, random forest, XGBoost, and K-nearest neighbors (KNN) will be applied to cluster patients into diabetic and non-diabetic. Its primary goals would be data preprocessing, metrics assessment of the model performance, and the choice of the most adequate model depending on the accuracy and interpretability.

To analyze the dataset of diabetes patients (that has several features, such as age, BMI, HbA1c, and blood glucose levels) for predicting diabetes prediction and doing this by comparing 4 models (Linear Regression, Random Forest, XGBoost and KNN) were used.  Classification performance were checked which results in Random Forest and KNN models outperformed others.

The results show that machine learning can effectively predict diabetes by using data. 
### Team members
- Hemlatha Kaur Saran [hsaran@sandiego.edu]
- Raminder Singh [ramindersingh@sandiego.edu]
- George David Asirvatharaj [gdavidasirvatharaj@sandiego.edu]

### Technologies

- [Python] - framework to use statistics lib and visualizations
- [VisualStudio Code] - OpenSource text editor
- [Jupyter Extension] - to build, manage and run notebooks
- [python libs] - jupyter notebook pandas numpy matplotlib scipy scikit-learn pandoc
nbconvert[webpdf] nbconvert notebook-as-pdf seaborn xgboost shap openpyxl
- [Chrome Browser] - to run the notebook and to convert to pdf
- [Github.com] - to maintain the code repo and to upload the final submissions
- [Zoom] - for collaborations and recording

### Project Description

**_Extracts taken from project report_**

Healthcare has been changed due to the inclusion of machine learning (ML) that has allowed predictive models to help diagnose and predict a disease. More specifically, ML algorithms are capable of detecting trends in complicated healthcare-related information, including medical records and patient history (Olalekan Kehinde, 2025). These models have been strong in disease prediction of such illnesses as diabetes, empowering early diagnosis, individualized treatment approaches, and patient outcomes, becoming one of the sources of change in medical practice across the world. 

This report aims to use machine learning to predict diabetes with the help of a dataset consisting of several health characteristics. Such models as logistic regression, random forest, XGBoost, and K-nearest neighbors (KNN) will be applied to cluster patients into diabetic and non-diabetic. Its primary goals would be data preprocessing, metrics assessment of the model performance, and the choice of the most adequate model depending on the accuracy and interpretability.

In the first phase of analysis, The histograms show graphically the number of occurrences of numerical data like age, BMI, HbA1c, and blood glucose levels. The age attribute has a maximum at 50 years, signifying that many of the people would be in middle age. The BMI shows a high peak of 30 that indicates more people are overweight or obese. The concentration of HbA1c level is maximized at 6 with some elevated measurements. The concentration of blood glucose has been highly concentrated between 100 and 150 mg/dL. The distribution of diabetes is very skewed, as the cases of non-diabetes are bigger in number.

HbA1c level also has a strong positive relationship with blood glucose level of 0.42, which shows that an increase in blood glucose level implies an increase in HbA1c level. The correlations between age and BMI with diabetes are also positive but moderate (0.26 and 0.21, respectively), which implies that diabetes is more likely to occur in older people and in people who have higher BMI. 

The dataset utilized in this study was an open public one related to the prediction of diabetes. It contains 100,000 rows and 8 columns featuring age, gender, the presence of hypertension, heart disease, body mass index (BMI), level of HbA1c, level of blood glucose, and history of smoking. The target variable is the presence or absence of diabetes (1) or not (0). The features act as predictors when determining the existence or the nonexistence of diabetes.

Logistic regression is an efficient but easy model of binary classification. It avails the interpretability of estimating the probability of an outcome on the basis of input features. In this diabetic prediction model, the logistic regression had a performance accuracy of 95.90%, a precision of 86.46%, and a recall of 61.71%, showing that although the model performs sufficiently in distinguishing between the non-diabetic instances, it does not perform well when it comes to recognizing the diabetic instances. F1 was 0.7202, and ROC AUC was 0.8040, which displays a satisfactory result.

The Random Forest algorithm also gives good clues as to which features are the most influential following the predictions of diabetes. The level of HbA1c and the level of blood glucose have the most significance, having the relative importance of 0.3971 and 0.3296, respectively, and thus are the most relevant ones in predicting diabetes. Other significant characteristics are the BMI (0.1220) and age (0.1004) that also significantly contribute towards the decisions made by the model. Such features as heart disease (0.0107) and gender (0.0000) are of very low importance as compared to the others.

XGBoost is a powerful model that runs on gradient boosting, and it is also great with imbalanced data. XGBoost presented an accuracy of 97.17 percent, a precision of 95.68 percent, and a recall of 69.96 percent in the given diabetes prediction problem. This showed a good precision versus recall balance F1 of 0.8083. The ROC AUC value of 0.8483 also shows that it is able to, in a strong way, separate the classes compared to having simpler models such as the logistic regression.

The performance of each model based on their evaluation results is impressive, with Random Forest ranking ahead, as it achieved an accuracy of 96.99% and a precision of 94.67, and XGBoost almost followed with an accuracy of 97.17% and a precision of 95.68. Both models have high recall scores, which are 69.96 percent and 68.68 percent, respectively, and thus demonstrate that they are effective in identifying positive cases of diabetics. Logistic regression with the accuracy of 95.90 also proved to be very accurate but had a lower recall, 61.71, and thus, it was not as good at identifying diabetic patients as the ensemble models. KNN was a good baseline, where it achieved a 96.07% rate of accuracy but with a low rate of recall of 61.07 and a rate of precision of 89.60.
```
Basic statistics for numerical features:
                 age  hypertension  heart_disease            bmi  \
count  100000.000000  100000.00000  100000.000000  100000.000000   
mean       41.885856       0.07485       0.039420      27.320767   
std        22.516840       0.26315       0.194593       6.636783   
min         0.080000       0.00000       0.000000      10.010000   
25%        24.000000       0.00000       0.000000      23.630000   
50%        43.000000       0.00000       0.000000      27.320000   
75%        60.000000       0.00000       0.000000      29.580000   
max        80.000000       1.00000       1.000000      95.690000   

         HbA1c_level  blood_glucose_level       diabetes  
count  100000.000000        100000.000000  100000.000000  
mean        5.527507           138.058060       0.085000  
std         1.070672            40.708136       0.278883  
min         3.500000            80.000000       0.000000  
25%         4.800000           100.000000       0.000000  
50%         5.800000           140.000000       0.000000  
75%         6.200000           159.000000       0.000000  
max         9.000000           300.000000       1.000000  
```

The histograms show graphically the number of occurrences of numerical data like age, BMI, HbA1c, and blood glucose levels. The age attribute has a maximum at 50 years, signifying that many of the people would be in middle age. The BMI shows a high peak of 30 that indicates more people are overweight or obese. The concentration of HbA1c level is maximized at 6 with some elevated measurements. The concentration of blood glucose has been highly concentrated between 100 and 150 mg/dL. The distribution of diabetes is very skewed, as the cases of non-diabetes are bigger in number.

Blood glucose and HbA1c levels are strongly correlated (0.42), Moderate correlation between diabetes, age (0.26) and BMI (0.21), Correlation analysis reveals key predictors for diabetes prediction Understanding relationships between features is crucial for model selection.

Logistic Regression is a binary classification model based on probability It is interpretable and provides insights into the relationship between features and diabetes
Suitable for linearly separable data        Simple and effective for baseline model comparison

Random Forest is an ensemble method that uses multiple decision trees It aggregates results from multiple trees for robust predictions The model is capable of capturing complex patterns in data It helps identify feature importance, providing insight into predictors

The XGBoost model correctly predicted 97.17% of the cases. The model correctly identified 95.68% of the positive predictions as diabetic cases. The recall of 69.96% indicates that the model correctly identified 69.96% of actual diabetic patients. An F1 score of 0.8083 shows a good balance between precision and recall. The ROC AUC of 0.8483 reflects the model’s strong ability to discriminate between diabetic and non-diabetic cases.

The KNN model correctly predicted 96.07% of the cases in the dataset. The model correctly identified 89.60% of the predicted diabetic cases. The recall of 61.07% indicates that the model identified 61.07% of the actual diabetic patients. An F1 score of 0.7263 reflects a moderate balance between precision and recall. The ROC AUC of 0.8020 shows the model’s ability to distinguish between diabetic and non-diabetic cases.

XGBoost has higher ROC AUC and Recall and good in distinguishing between the two classes and is also the most effective at identifying the actual positive cases (diabetic patients). Random Forest Tuned is the next closest to have a high ROC AUC and Recall

For full details refer to [AAI-501-IN2-Group7-project-final.pdf]

### License
MIT
- Top-level fiter expression: _owner:ramindersinghusd license:MIT_

### Acknowledgments
We as Team-4 members really thankful to Prof Azka A for her support, guidance and  making it some easily for us to understand the Probability and Statistics fundamentals throughout this module - AAI-501-IN2
