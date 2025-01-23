import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Part A

# Read our data, create the dataframe
df1 = pd.read_excel(r"c:\Users\Γιωργος\Desktop\MECHANICAL ENGINEERING\4ο ΕΤΟΣ\ΑΝΑΛΥΣΗ ΔΕΔΟΜΕΝΩΝ\1η εργασια\Data Analysis_2024 1st Case_Data.xlsx")
print(df1.head())
print(df1.describe())
print(df1.info())

# Check for NaN values
print(df1.isnull().values.any()) #prepei na vgazei False, an den exw NaN values se olo to dataframe

# Remove years 2007,2013 from the dataframe 
i = df1[(df1['Year']==2013) | (df1['Year']==2007)].index
print(i)
df1.drop(index=i,inplace=True)
print(df1.info()) #gia epalitheysh, se sygrish me to prohgoumeno info

# Get the distribution plot of our target-variable
print(sns.histplot(data=df1['Life expectancy '],stat='density',kde=True))
plt.show()

predictors = ['Year', 'Adult Mortality','Alcohol', 'percentage expenditure', 
              'Hepatitis B', 'Measles ', ' BMI ','under-five deaths ', 'Polio', 'Total expenditure',
              'Diphtheria ',' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
              'Income composition of resources', 'Schooling']

# Define the y,X for the first model based on the 1st project
y_m1 = df1['Life expectancy ']
X_m1 = df1[['Year', 'Adult Mortality','Alcohol', ' BMI ','under-five deaths ',
              'Diphtheria ',' HIV/AIDS','Income composition of resources', 'Schooling']]

# Define the y,X for the second(forward selection) model based on the 1st project
y_m2 = df1['Life expectancy ']
X_m2 = df1[['Year', 'Adult Mortality','Alcohol', 'percentage expenditure', 
              'Hepatitis B', 'Measles ', ' BMI ','under-five deaths ', 'Polio', 'Total expenditure',
              'Diphtheria ',' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
              'Income composition of resources', 'Schooling']]

m1 = LinearRegression()
m2 = LinearRegression()

test_mse_m1 = []
test_mse_m2 = []

# k-fold cross-validation on the first model
kf1 = KFold(n_splits=5, shuffle=True, random_state=33)
for train_index, test_index in kf1.split(X_m1,y_m1):
    X_train, X_test = X_m1.values[train_index], X_m1.values[test_index]
    y_train, y_test = y_m1.values[train_index], y_m1.values[test_index]
    m1.fit(X_train,y_train)
    y_pred = m1.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    test_mse_m1.append(mse)

# k-fold cross-validation on the second model
kf2 = KFold(n_splits=5, shuffle=True, random_state=33)
for train_index, test_index in kf2.split(X_m2,y_m2):
    X_train, X_test = X_m2.values[train_index], X_m2.values[test_index]
    y_train, y_test = y_m2.values[train_index], y_m2.values[test_index]
    m2.fit(X_train,y_train)
    y_pred = m2.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    test_mse_m2.append(mse)

# Calculate each model's test MSE
avg_mse_m1 = sum(test_mse_m1) / len(test_mse_m1)
avg_mse_m2 = sum(test_mse_m2) / len(test_mse_m2)
print("test MSE of the 1st model is:",avg_mse_m1)
print("test MSE of the 2nd(forward selection) model is:",avg_mse_m2)

# Leave-One-Out cross-validation for the 2nd model
test_mse_m2_loo = []
loo = LeaveOneOut()
for train_index, test_index in loo.split(X_m2):
    X_train, X_test = X_m2.values[train_index], X_m2.values[test_index]
    y_train, y_test = y_m2.values[train_index], y_m2.values[test_index]
    m2.fit(X_train, y_train)
    y_pred = m2.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    test_mse_m2_loo.append(mse)
    
avg_mse_m2_loo = np.mean(test_mse_m2_loo)
print("test MSE of the 2nd(forward selection) model with the loo method is:",avg_mse_m2_loo)

# Part B 
print(df1.head())
print(df1.tail())
print(df1.info())

# Convert Life expectancy into a dummy variable
avg_life_expectancy = df1['Life expectancy '].mean()
print(avg_life_expectancy)
life_expectancy_dummies = pd.get_dummies(df1['Life expectancy '] > avg_life_expectancy ,
                                        prefix='Life expectancy ', drop_first=True)
life_expectancy_dummies = life_expectancy_dummies.astype(int)
print(life_expectancy_dummies)
print(life_expectancy_dummies.value_counts()) #epalitheysh
df2 = pd.concat([df1, life_expectancy_dummies], axis=1)
df2.drop(['Life expectancy ','Year','Country', 'Status'], axis=1, inplace=True)
print(df2.head())
print(df2.info())
print(df2.columns)

# to df2 einai pleon etoimo gia ton ML algorithmo

# Logistic Regression 
y_m3 = df2['Life expectancy _True'] 
X_m3 = df2[['Adult Mortality', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling',]]

X3_train, X3_test, y3_train, y3_test = train_test_split(X_m3, y_m3, test_size=0.2, random_state=33)
logreg_model = LogisticRegression()
logreg_model.fit(X3_train,y3_train)
print("Intercept of the logistic regression model:", logreg_model.intercept_[0])
print("Coefficients of the logistic regression model:")
for i, coef in enumerate(logreg_model.coef_[0]):
    print(f"Coefficient of {X_m3.columns[i]}: {coef}")

y3_pred = logreg_model.predict(X3_test)
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix(y3_test, y3_pred))

train_acc_logreg = accuracy_score(y3_train, logreg_model.predict(X3_train))
print("Training accuracy of the logistic regression model:", train_acc_logreg)

acc05 = accuracy_score(y3_test,y3_pred)
print("accuracy on test data with 0.5 threshold is:",acc05)
print("classification report for threshold=0.5:")
print(classification_report(y3_test,y3_pred))

y3_pred_prob = logreg_model.predict_proba(X3_test)[:, 1]

# Apply threshold=0.4
y3_pred_04 = (y3_pred_prob > 0.4).astype(int)
acc04 = accuracy_score(y3_test,y3_pred_04)
print("accuracy on test data with 0.4 threshold is:",acc04)
print("classification report for threshold=0.4:")
print(classification_report(y3_test,y3_pred_04))

# Apply threshold=0.6
y3_pred_06 = (y3_pred_prob > 0.6).astype(int)
acc06 = accuracy_score(y3_test,y3_pred_06)
print("accuracy on test data with 0.6 threshold is:",acc06)
print("classification report for threshold=0.6:")
print(classification_report(y3_test,y3_pred_06))

# LDA 
y_m4 = df2['Life expectancy _True'] 
X_m4 = df2[['Adult Mortality', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling',]]

X4_train, X4_test, y4_train, y4_test = train_test_split(X_m4, y_m4, test_size=0.2, random_state=33)
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X4_train, y4_train)
print("Intercept of the LDA model:", lda_model.intercept_[0])
print("Coefficients of the LDA model:")
for i, coef in enumerate(lda_model.coef_[0]):
    print(f"Coefficient of {X_m4.columns[i]}: {coef}")

train_acc_lda = accuracy_score(y4_train, lda_model.predict(X4_train))
print("Training accuracy of the LDA model:", train_acc_lda)

y4_pred = lda_model.predict(X4_test)
acc_lda = accuracy_score(y4_test, y4_pred)
print("\nConfusion Matrix for LDA:")
print(confusion_matrix(y4_test, y4_pred))
print("accuracy on test data with LDA model is:",acc_lda)
print("classification report for LDA model:")
print(classification_report(y4_test,y4_pred))

# KNN
y_m5 = df2['Life expectancy _True'] 
X_m5 = df2[['Adult Mortality', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling',]]
X5_train, X5_test, y5_train, y5_test = train_test_split(X_m5, y_m5, test_size=0.2, random_state=33)
k_values = [1,3,5,7,9]
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X5_train,y5_train)
    y5_pred = knn_model.predict(X5_test)
    acc_knn = accuracy_score(y5_test,y5_pred)
    print(f"accuracy on test data with K={k} is: {acc_knn}")
    print(f"classification report for KNN with K={k}:")
    print(classification_report(y5_test,y5_pred))
    print(f"\nConfusion Matrix for KNN with k={k}:")
    print(confusion_matrix(y5_test, y5_pred))

