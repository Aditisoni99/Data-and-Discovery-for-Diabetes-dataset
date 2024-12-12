#installing libraries
library (dplyr)
library (ggplot2)
#loading the data 
df_csv <- read.csv("C:\\Users\\aditi\\Downloads\\diabetes.csv")

# Count null values in each column
sapply(df_csv, function(x) sum(is.na(x)))

# count missing values in each column
# Replace zeros with NA for relevant columns
columns_to_clean <- c("Pregnancies", "Age")
df_csv[columns_to_clean] <- lapply(df_csv[columns_to_clean], function(x) ifelse(x == 0, NA, x))

# Impute missing values with median
for (col in columns_to_clean) 
head(df_csv)

#summary of the dataset
summary(df_csv)

# Count the Outcome variable (0 = No Diabetes, 1 = Diabetes)
outcome_counts <- table(df_csv$Outcome)

# Convert to a data frame for plotting
outcome_df <- as.data.frame(outcome_counts)
colnames(outcome_df) <- c("Outcome", "Count")
outcome_df$Outcome <- factor(outcome_df$Outcome, labels = c("No Diabetes", "Diabetes"))

# Plot the pie chart using ggplot2
ggplot(outcome_df, aes(x = "", y = Count, fill = Outcome)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  theme_void() + 
  labs(title = "Diabetes Outcome pie chart") +
  scale_fill_manual(values = c("blue", "green"))

#descriptive statistics
descriptive_statistics <- df_csv %>%
  summarize(
Pregnancy_mean <- mean(Pregnancies, na.rm = TRUE),
Pregnancy_median <- median(Pregnancies, na.rm = TRUE),
Pregnancy_Std <- sd(Pregnancies, na.rm = TRUE),
Pregnancy_min<- min(Pregnancies, na.rm = TRUE),
Pregnancy_max <- max(Pregnancies, na.rm = TRUE),

Age_mean <- mean(Age, na.rm = TRUE),
Age_median <- median(Age, na.rm = TRUE),
Age_std <- sd(Age, na.rm = TRUE),
Age_min <- min(Age, na.rm = TRUE),
Age_max <- max(Age, na.rm = TRUE), 
)
print (descriptive_statistics)
# Convert Outcome to a factor with labels "No" and "Yes"
df_csv$Outcome <- factor(df_csv$Outcome, labels = c("No", "Yes"))

#plotting the histogram 
# Convert Outcome to a factor with labels "No" and "Yes"
df_csv$Outcome <- factor(df_csv$Outcome, labels = c("No", "Yes"))

# Histogram for Pregnancies with Outcome labels
ggplot(df_csv, aes(x = Pregnancies, fill = Outcome)) +
  geom_histogram(binwidth = 1, color = "black", position = "dodge") +
  theme_minimal() +
  labs(title = "Histogram of Pregnancies by Diabetes Outcome", x = "Number of Pregnancies", y = "Frequency") +
  scale_fill_manual(values = c("pink", "green"), name = "Diabetes Outcome")

# Histogram for Age with Outcome labels
ggplot(df_csv, aes(x = Age, fill = Outcome)) +
  geom_histogram(binwidth = 2, color = "black", position = "dodge") +
  theme_minimal() +
  labs(title = "Histogram of Age by Diabetes Outcome", x = "Age", y = "Frequency") +
  scale_fill_manual(values = c("pink", "purple"), name = "Diabetes Outcome")

# boxes colored for ease of interpretation
boxplot(Pregnancies~ Outcome, data= df_csv, notch=TRUE,
        col=(c("red")),
        main="Boxplot for pregnancies", xlab="pregnancies")

boxplot(Age~ Outcome, data= df_csv, notch=TRUE,
        col=(c("blue")),
        main="Boxplot for Age", xlab="age")

#plotting the correlogram 
# install.packages ("ggcorrplot")
# library(ggcorrplot)

#correlation matrix for all the variables
correlation_matrix <- cor(df_csv[sapply(df_csv, is.numeric)], use= "complete.obs")
print (correlation_matrix)

corr_data <- df_csv %>% select(Pregnancies, Age,Glucose, BloodPressure, SkinThickness ,BMI, DiabetesPedigreeFunction, Outcome)
corr_data$Outcome <- as.numeric(corr_data$Outcome)

correlation_matrix <- cor(corr_data)
print(correlation_matrix)
ggcorrplot(correlation_matrix, method = "circle", type ="lower",lab = TRUE, colors = c("blue"), 
            title = "correlogram of Pregnancies, Age, and Outcome of Diabetes", lab_size= 4)

#inferential Statistics 
T_test_Age <- t.test(Age ~ Outcome, data= df_csv)
print(T_test_Age)

T_test_Pregnancies <- t.test(Pregnancies ~ Outcome, data= df_csv)
print(T_test_Pregnancies)

#Anova test 
anova_age <- aov(Age ~ Outcome, data = df_csv)
summary (anova_age)
anova_preg <- aov(Pregnancies ~ Outcome, data = df_csv)
summary (anova_preg)

# #machine learning and inferential statistics 
#installing libraries

library(caret)
#splitting the data set 
#train and testing data 
df_csv <- df_csv[complete.cases(df_csv),]
trainIndex <- createDataPartition(df_csv$Outcome, p= .8, list = FALSE) 
train_data <- df_csv[trainIndex,]
test_data <- df_csv[- trainIndex,]
dim(train_data)
dim(test_data)

#training the model using cross validation
# set.seed (50)
model_cv <- trainControl(method = "cv", number = 5, savePredictions = TRUE)

#random forest model 
# set.seed(50)
randomforest_model <- train(Outcome ~ ., data= train_data, method= "rf",trControl = model_cv)

#predictions 
prediction_rf <- predict(randomforest_model, newdata = test_data) 
prediction_rf <- droplevels(prediction_rf)

#evaluating the model 
conf_matrix_rf <- confusionMatrix(data = prediction_rf, reference = factor(test_data$Outcome)) 
conf_matrix_pre_recall_rf <- confusionMatrix(data = prediction_rf, reference = factor(test_data$Outcome), mode = "prec_recall")
summary (randomforest_model)
print(conf_matrix_rf)
fourfoldplot(as.table(conf_matrix_rf),color=c("yellow","pink"),main = "Confusion Matrix RF")
print(conf_matrix_pre_recall_rf)

#printing the results
print (randomforest_model)
importance_rf <- varImp(randomforest_model, scale = FALSE)
print (importance_rf)

#logistic regression model
set.seed(50)
logReg_model <- train(Outcome ~ ., data=train_data, method= "regLogistic", trControl = model_cv)

#predictions 
prediction_logReg = predict(logReg_model, newdata = test_data)

#evaluating the model
conf_matrix_logReg <- confusionMatrix(data = prediction_logReg, reference = factor(test_data$Outcome)) 
conf_matrix_logReg_prec_recall <- confusionMatrix(data = prediction_logReg, reference = factor(test_data$Outcome), mode = "prec_recall")
print(conf_matrix_logReg)
fourfoldplot(as.table(conf_matrix_logReg),color=c("yellow","pink"),main = "Confusion Matrix logReg")
print(conf_matrix_logReg_prec_recall)

#printing the results
print (logReg_model)
importance_logReg <- varImp(logReg_model, scale = FALSE)
print (importance_rf)

#leveling the outcome
test_data$Outcome <- factor(test_data$Outcome, levels = c("No", "Yes"))

# Logistic Regression Model ROC and AUC
predicted_probs_logReg <- predict(logReg_model, newdata = test_data, type = "prob")[, 2]  # Extract probabilities for the "Yes" outcome

# Calculate the ROC curve
roc_logreg <- roc(test_data$Outcome, predicted_probs_logReg)

# Plot the ROC curve
plot(roc_logreg, col = "green", lwd = 2, main = "ROC Curve - Logistic Regression")
abline(a = 0, b = 1, col = "gray", lty = 2)  # Diagonal line for random model

# Display the AUC value
auc_logreg <- auc(roc_logreg)
cat("AUC for Logistic Regression:", auc_logreg, "\n")

# Random Forest Model ROC and AUC
predicted_probs_rf <- predict(randomforest_model, newdata = test_data, type = "prob")[, 2]  # Extract probabilities for the "Yes" outcome

#ROC curve
roc_rf <- roc(test_data$Outcome, predicted_probs_rf)

# Plot the ROC curve for Random Forest
plot(roc_rf, col = "blue", lwd = 2, main = "ROC Curve - Random Forest")
abline(a = 0, b = 1, col = "gray", lty = 2)  

# Display the AUC value
auc_rf <- auc(roc_rf)
cat("AUC for Random Forest:", auc_rf, "\n")

# Combined ROC Curves for Comparison
plot(roc_logreg, col = "green", lwd = 2, main = "ROC Curve Comparison")
lines(roc_rf, col = "blue", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col = c("green", "blue"), lwd = 2)


