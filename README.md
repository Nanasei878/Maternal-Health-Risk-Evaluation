# Maternal-Health-Risk-Evaluation
 Which health conditions are the strongest indications for health risks during pregnancy?
Project Description: Maternal Health Risk Analysis
Objective:
The primary objective of this project is to analyze and predict maternal health risk levels using a dataset collected from various hospitals, community clinics, and maternal health care centers through an IoT-based risk monitoring system. The project aims to identify key health indicators that contribute to maternal health risks and develop a predictive model to categorize the risk levels of pregnant women.

Dataset:
The dataset consists of the following features:

Age: Age of the pregnant woman in years.
SystolicBP: Systolic blood pressure in mmHg.
DiastolicBP: Diastolic blood pressure in mmHg.
BS: Blood glucose levels in mmol/L.
BodyTemp: Body temperature in degrees Fahrenheit.
HeartRate: Heart rate in beats per minute.
RiskLevel: Predicted risk intensity level during pregnancy (categorized as low risk, mid risk, and high risk).
Steps Undertaken:
Data Loading and Initial Inspection:

Loaded the dataset into a pandas DataFrame.
Inspected the dataset for structure, summary statistics, and missing values.
Exploratory Data Analysis (EDA):

Visualized the distribution of each numeric feature using histograms, box plots, and swarm plots.
Created correlation heatmaps to understand the relationships between different features.
Data Preprocessing:

Split the data into training and testing sets.
Standardized the feature variables to ensure normal distributions and suppress outliers.
Encoded the categorical target variable (RiskLevel) using LabelEncoder.
Model Building and Evaluation:

Trained a RandomForestClassifier model on the training data.
Evaluated the model using accuracy, precision, recall, and F1-score metrics.
Visualized the confusion matrix to understand misclassifications.
Hyperparameter Tuning:

Performed hyperparameter tuning using GridSearchCV to optimize the RandomForestClassifier.
Compared the performance of the initial model with the optimized model using classification reports and confusion matrices.
Feature Importance:

Identified and visualized the importance of each feature in predicting maternal health risk levels using the RandomForestClassifier.
Key Findings:
Blood Glucose Levels (BS): Identified as the most important feature in predicting maternal health risks.
Systolic Blood Pressure (SystolicBP): Also a significant predictor of risk levels.
Age: An important factor, especially in distinguishing high-risk pregnancies.
Performance: The initial RandomForestClassifier model achieved an accuracy of 0.88, while the optimized model had a slightly lower accuracy of 0.83.
Conclusion:
This project successfully analyzed maternal health risk factors and developed a predictive model to classify risk levels. The findings highlight the importance of monitoring blood glucose levels and blood pressure in pregnant women to assess and manage potential health risks. The project demonstrates the value of using machine learning techniques in healthcare to improve maternal health outcomes.

Future Work:
Model Improvement: Further optimization of the model using a more extensive hyperparameter grid and additional algorithms.
Feature Engineering: Exploration of new features and interaction terms to improve predictive performance.
Real-time Monitoring: Integration with IoT devices for real-time risk assessment and intervention.
This project serves as a comprehensive example of using data analysis and machine learning to address critical healthcare challenges and improve patient outcomes.
