# Data-and-Discovery-for-Diabetes-dataset
This project leverages statistical analysis and machine learning to predict the likelihood of diabetes based on a set of health-related features from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) dataset. 
The focus of this analysis is to explore key predictors, such as pregnancy and age, and develop predictive models to help identify individuals at risk of diabetes.

# Key Steps and Methodology:
# Data Exploration and Preprocessing:

The dataset contains features like age, number of pregnancies, blood pressure, glucose levels, and more.
Performed data cleaning, including handling missing values, normalizing data, and ensuring consistency for the predictive models.
Conducted exploratory data analysis (EDA) to identify key correlations and trends in features such as pregnancy count and age, which are significant predictors of diabetes.

# Model Development:

Built Random Forest and Logistic Regression models to predict the presence of diabetes.
Random Forest: A robust ensemble method known for handling complex datasets by creating multiple decision trees and aggregating their results.
Logistic Regression: A simpler model useful for understanding the relationship between input variables and the binary outcome (diabetes or not).

# Model Evaluation:

Cross-validation was used to ensure the models generalize well to unseen data.
Performance metrics like accuracy, precision, recall, and F1-score were calculated to evaluate model performance and ensure reliable diabetes predictions.

# Results:

Random Forest achieved an accuracy of 76.47% and Logistic Regression achieved 75.82% accuracy.
These models were assessed with precision, recall, and F1-score to balance false positives and negatives, ensuring reliable predictions.
# Key Features:
Pregnancy and Age were found to be significant predictors of diabetes risk.
Models trained and evaluated in R with the use of libraries such as caret, randomForest, and glm for model implementation and evaluation.
# Data visualization with ggplot2 was used to plot feature distributions and model evaluation metrics.
# Technologies Used:
R: Statistical analysis, machine learning model development, and evaluation.
caret: For data splitting, model training, and evaluation.
randomForest: For implementing the Random Forest model.
glm: For implementing Logistic Regression.
ggplot2: For visualizations.

# Future Work:
Experiment with additional machine learning algorithms (e.g., Support Vector Machine, XGBoost) to improve prediction accuracy.
Perform hyperparameter tuning to enhance model performance further.
Explore the integration of more health-related features or external datasets to improve the robustness and accuracy of the diabetes prediction models.
