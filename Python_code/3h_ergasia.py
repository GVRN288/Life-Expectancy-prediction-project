import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

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


y = df2['Life expectancy _True'] 
X = df2[['Adult Mortality', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling',]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Decision Tree
m1 = DecisionTreeClassifier(max_depth=3)
m1.fit(X_train,y_train)
m1_pred = m1.predict(X_test)
m1_acc = accuracy_score(y_test, m1_pred)
print("Decision Tree test accuracy is:", m1_acc)
print("\n")
print(classification_report(y_test,m1_pred))

# Bagging
m2 = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=200, random_state=101)
m2.fit(X_train,y_train)
m2_pred = m2.predict(X_test)
m2_acc = accuracy_score(y_test, m2_pred)
print("Bagging test accuracy is:", m2_acc)
print("\n")
print(classification_report(y_test,m2_pred))

# Random Forest
m3 =RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=101)
m3.fit(X_train,y_train)
m3_pred = m3.predict(X_test)
m3_acc = accuracy_score(y_test,m3_pred)
print("Random Forest accuracy is:", m3_acc)
print("\n")
print(classification_report(y_test,m3_pred))

# Boosting
m4 = AdaBoostClassifier(n_estimators=200, learning_rate=1, algorithm="SAMME", random_state=101)
m4.fit(X_train,y_train)
m4_pred = m4.predict(X_test)
m4_acc = accuracy_score(y_test,m4_pred)
print("Boosting accuracy is:", m4_acc)
print("\n")
print(classification_report(y_test,m4_pred))

# test error vs depth for decision tree
depths = range(1,31)
test_errors1 = []
for md in depths:
       tree_model = DecisionTreeClassifier(max_depth=md, random_state=101)
       tree_model.fit(X_train,y_train)
       tree_pred = tree_model.predict(X_test)
       tree_test_error = 1 - accuracy_score(y_test,tree_pred)
       test_errors1.append(tree_test_error)
       
plt.figure(figsize=(10,7))
plt.plot(depths, test_errors1, marker='o', markerfacecolor='red')
plt.title('Test Error vs Depth of Decision Tree')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Test Error')
plt.show()

# test error vs n_estimators for bagging
test_errors2 = []
n_estim = range(10,500,10)
for n in n_estim:
       bagg = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=n, random_state=101)
       bagg.fit(X_train,y_train)
       bagg_pred = bagg.predict(X_test)
       bagg_test_error = 1 - accuracy_score(y_test, bagg_pred)
       test_errors2.append(bagg_test_error)
  
plt.figure(figsize=(10,7))
plt.plot(n_estim, test_errors2, marker='o', markerfacecolor='red')
plt.title('Test Error vs Number of base estimators for Bagging')
plt.xlabel('Number of base estimators')
plt.ylabel('Test Error')
plt.show()

# test error vs n_estimators for Random Forest
test_errors3 = []
for n in n_estim:
       rf =RandomForestClassifier(n_estimators=n, max_features='sqrt', random_state=101)
       rf.fit(X_train,y_train)
       rf_pred = rf.predict(X_test)
       rf_test_error = 1 - accuracy_score(y_test, rf_pred)
       test_errors3.append(rf_test_error)
       
plt.figure(figsize=(10,7))
plt.plot(n_estim, test_errors3, marker='o', markerfacecolor='red')
plt.title('Test Error vs Number of base estimators for Random Forest')
plt.xlabel('Number of base estimators')
plt.ylabel('Test Error')
plt.show()

# test error vs n_estimators for boosting
test_errors4 = []
for n in n_estim:
       boo = AdaBoostClassifier(n_estimators=n, learning_rate=1, algorithm="SAMME", random_state=101)
       boo.fit(X_train, y_train)
       boo_pred = boo.predict(X_test)
       boo_test_error = 1 - accuracy_score(y_test, boo_pred)
       test_errors4.append(boo_test_error)
       
plt.figure(figsize=(10,7))
plt.plot(n_estim, test_errors4, marker='o', markerfacecolor='red')
plt.title('Test Error vs Number of base estimators for Boosting')
plt.xlabel('Number of base estimators')
plt.ylabel('Test Error')
plt.show()

all_test_acc = [m1_acc, m2_acc, m3_acc, m4_acc]
methods = ['Decision Tree', 'Bagging', 'Random Forest', 'Boosting']
results = pd.DataFrame(data=all_test_acc, index=methods, columns=['test accuracy'])
print(results)

