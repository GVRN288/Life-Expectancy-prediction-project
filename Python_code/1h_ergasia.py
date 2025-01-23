#1h ergasia
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

# Read our data, create the dataframe
df1 = pd.read_excel(r"c:\Users\Γιωργος\Desktop\MECHANICAL ENGINEERING\4ο ΕΤΟΣ\ΑΝΑΛΥΣΗ ΔΕΔΟΜΕΝΩΝ\1η εργασια\Data Analysis_2024 1st Case_Data.xlsx")
print(df1.head())
#print(df1.describe())
print(df1.info())

# Check for NaN values
print(df1.isnull().values.any()) #prepei na vgazei False, an den exw NaN values se olo to dataframe

# Remove years 2007,2013 from the dataframe 
i = df1[(df1['Year']==2013) | (df1['Year']==2007)].index
print(i)
df1.drop(index=i,inplace=True)
print(df1.info())

# Get the distribution plot of our target-variable
print(sns.histplot(data=df1['Life expectancy '],stat='density',kde=True))
plt.show()

print(df1.columns)
df1cr = df1.corr(numeric_only=True)
print(df1cr.loc['Life expectancy '])
predictors = ['Year', 'Adult Mortality','Alcohol', 'percentage expenditure', 
              'Hepatitis B', 'Measles ', ' BMI ','under-five deaths ', 'Polio', 'Total expenditure',
              'Diphtheria ',' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
              'Income composition of resources', 'Schooling']

# Life expectancy versus Year
y = df1['Life expectancy ']
y= y.values.reshape(-1,1)
X1 = df1['Year']
X1 = X1.values.reshape(-1,1)
sns.lmplot(data=df1,x='Year',y='Life expectancy ')
plt.show()


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=99)
lm1 = LinearRegression()
lm1.fit(X1_train,y1_train) 
prd1 = lm1.predict(X1_test)
MAE1 = mean_absolute_error(y1_test,prd1)
MSE1 = mean_squared_error(y1_test,prd1)
RMSE1 = np.sqrt(MSE1)
R2sc1 = r2_score(y1_test,prd1)
est1 = sm.OLS
#print(plt.scatter(y1_test,prd1))
#plt.show()

# Life expectancy versus Adult Mortality
X2 = df1['Adult Mortality']
X2 = X2.values.reshape(-1,1)
sns.lmplot(data=df1,x='Adult Mortality',y='Life expectancy ')
plt.show()
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=99)
lm2 = LinearRegression()
lm2.fit(X2_train,y2_train) 
prd2 = lm2.predict(X2_test)
MAE2 = mean_absolute_error(y2_test,prd2)
MSE2 = mean_squared_error(y2_test,prd2)
RMSE2 = np.sqrt(MSE2)
R2sc2 = r2_score(y2_test,prd2)
#print(plt.scatter(y2_test,prd2))
#plt.show()

# Life expectancy versus Alcohol
X3 = df1['Alcohol']
X3 = X3.values.reshape(-1,1)
sns.lmplot(data=df1,x='Alcohol',y='Life expectancy ')
plt.show()
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.3, random_state=99)
lm3 = LinearRegression()
lm3.fit(X3_train,y3_train) 
prd3 = lm3.predict(X3_test)
MAE3 = mean_absolute_error(y3_test,prd3)
MSE3 = mean_squared_error(y3_test,prd3)
RMSE3 = np.sqrt(MSE3)
R2sc3 = r2_score(y3_test,prd3)
#print(plt.scatter(y3_test,prd3))
#plt.show()

# Life expectancy versus percentage expenditure
X4 = df1['percentage expenditure']
X4 = X4.values.reshape(-1,1)
sns.lmplot(data=df1,x='percentage expenditure',y='Life expectancy ')
plt.show()
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y, test_size=0.3, random_state=99)
lm4 = LinearRegression()
lm4.fit(X4_train,y4_train) 
prd4 = lm4.predict(X4_test)
MAE4 = mean_absolute_error(y4_test,prd4)
MSE4 = mean_squared_error(y4_test,prd4)
RMSE4 = np.sqrt(MSE4)
R2sc4 = r2_score(y4_test,prd4)
#print(plt.scatter(y4_test,prd4))
#plt.show()

# Life expectancy versus Hepatitis B
X5 = df1['Hepatitis B']
X5 = X5.values.reshape(-1,1)
sns.lmplot(data=df1,x='Hepatitis B',y='Life expectancy ')
plt.show()
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y, test_size=0.3, random_state=99)
lm5 = LinearRegression()
lm5.fit(X5_train,y5_train) 
prd5 = lm5.predict(X5_test)
MAE5 = mean_absolute_error(y5_test,prd5)
MSE5 = mean_squared_error(y5_test,prd5)
RMSE5 = np.sqrt(MSE5)
R2sc5 = r2_score(y5_test,prd5)
#print(plt.scatter(y5_test,prd5))
#plt.show()

# Life expectancy versus Measles 
X6 = df1['Measles ']
X6 = X6.values.reshape(-1,1)
sns.lmplot(data=df1,x='Measles ',y='Life expectancy ')
plt.show()
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y, test_size=0.3, random_state=99)
lm6 = LinearRegression()
lm6.fit(X6_train,y6_train) 
prd6 = lm6.predict(X6_test)
MAE6 = mean_absolute_error(y6_test,prd6)
MSE6 = mean_squared_error(y6_test,prd6)
RMSE6 = np.sqrt(MSE6)
R2sc6 = r2_score(y6_test,prd6)
#print(plt.scatter(y6_test,prd6))
#plt.show()

# Life expectancy versus  BMI 
X7 = df1[' BMI ']
X7 = X7.values.reshape(-1,1)
sns.lmplot(data=df1,x=' BMI ',y='Life expectancy ')
plt.show()
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y, test_size=0.3, random_state=99)
lm7 = LinearRegression()
lm7.fit(X7_train,y7_train) 
prd7 = lm7.predict(X7_test)
MAE7 = mean_absolute_error(y7_test,prd7)
MSE7 = mean_squared_error(y7_test,prd7)
RMSE7 = np.sqrt(MSE7)
R2sc7 = r2_score(y7_test,prd7)
#print(plt.scatter(y7_test,prd7))
#plt.show()

# Life expectancy versus under-five deaths 
X8 = df1['under-five deaths ']
X8 = X8.values.reshape(-1,1)
sns.lmplot(data=df1,x='under-five deaths ',y='Life expectancy ')
plt.show()
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y, test_size=0.3, random_state=99)
lm8 = LinearRegression()
lm8.fit(X8_train,y8_train) 
prd8 = lm8.predict(X8_test)
MAE8 = mean_absolute_error(y8_test,prd8)
MSE8 = mean_squared_error(y8_test,prd8)
RMSE8 = np.sqrt(MSE8)
R2sc8 = r2_score(y8_test,prd8)
#print(plt.scatter(y8_test,prd8))
#plt.show()

# Life expectancy versus Polio

X9 = df1['Polio']
X9 = X9.values.reshape(-1,1)
sns.lmplot(data=df1,x='Polio',y='Life expectancy ')
plt.show()
X9_train, X9_test, y9_train, y9_test = train_test_split(X9, y, test_size=0.3, random_state=99)
lm9 = LinearRegression()
lm9.fit(X9_train,y9_train) 
prd9 = lm9.predict(X9_test)
MAE9 = mean_absolute_error(y9_test,prd9)
MSE9 = mean_squared_error(y9_test,prd9)
RMSE9 = np.sqrt(MSE9)
R2sc9 = r2_score(y9_test,prd9)
#print(plt.scatter(y9_test,prd9))
#plt.show()

# Life expectancy versus Total expenditure

X10 = df1['Total expenditure']
X10 = X10.values.reshape(-1,1)
sns.lmplot(data=df1,x='Total expenditure',y='Life expectancy ')
plt.show()
X10_train, X10_test, y10_train, y10_test = train_test_split(X10, y, test_size=0.3, random_state=99)
lm10 = LinearRegression()
lm10.fit(X10_train,y10_train) 
prd10 = lm10.predict(X10_test)
MAE10 = mean_absolute_error(y10_test,prd10)
MSE10 = mean_squared_error(y10_test,prd10)
RMSE10 = np.sqrt(MSE10)
R2sc10 = r2_score(y10_test,prd10)
#print(plt.scatter(y10_test,prd10))
#plt.show()

# Life expectancy versus Diphtheria 
X11 = df1['Diphtheria ']
X11 = X11.values.reshape(-1,1)
sns.lmplot(data=df1,x='Diphtheria ',y='Life expectancy ')
plt.show()
X11_train, X11_test, y11_train, y11_test = train_test_split(X11, y, test_size=0.3, random_state=99)
lm11 = LinearRegression()
lm11.fit(X11_train,y11_train) 
prd11 = lm11.predict(X11_test)
MAE11 = mean_absolute_error(y11_test,prd11)
MSE11 = mean_squared_error(y11_test,prd11)
RMSE11 = np.sqrt(MSE11)
R2sc11 = r2_score(y11_test,prd11)
#print(plt.scatter(y11_test,prd11))
#plt.show()

# Life expectancy versus HIV/AIDS
X12 = df1[' HIV/AIDS']
X12 = X12.values.reshape(-1,1)
sns.lmplot(data=df1,x=' HIV/AIDS',y='Life expectancy ')
plt.show()
X12_train, X12_test, y12_train, y12_test = train_test_split(X12, y, test_size=0.3, random_state=99)
lm12 = LinearRegression()
lm12.fit(X12_train,y12_train) 
prd12 = lm12.predict(X12_test)
MAE12 = mean_absolute_error(y12_test,prd12)
MSE12 = mean_squared_error(y12_test,prd12)
RMSE12 = np.sqrt(MSE12)
R2sc12 = r2_score(y12_test,prd12)
#print(plt.scatter(y12_test,prd12))
#plt.show()

# Life expectancy versus GDP
X13 = df1['GDP']
X13 = X13.values.reshape(-1,1)
sns.lmplot(data=df1,x='GDP',y='Life expectancy ')
plt.show()
X13_train, X13_test, y13_train, y13_test = train_test_split(X13, y, test_size=0.3, random_state=99)
lm13 = LinearRegression()
lm13.fit(X13_train,y13_train) 
prd13 = lm13.predict(X13_test)
MAE13 = mean_absolute_error(y13_test,prd13)
MSE13 = mean_squared_error(y13_test,prd13)
RMSE13 = np.sqrt(MSE13)
R2sc13 = r2_score(y13_test,prd13)
#print(plt.scatter(y13_test,prd13))
#plt.show()

# Life expectancy versus Population
X14 = df1['Population']
X14 = X14.values.reshape(-1,1)
sns.lmplot(data=df1,x='Population',y='Life expectancy ')
plt.show()
X14_train, X14_test, y14_train, y14_test = train_test_split(X14, y, test_size=0.3, random_state=99)
lm14 = LinearRegression()
lm14.fit(X14_train,y14_train) 
prd14 = lm14.predict(X14_test)
MAE14 = mean_absolute_error(y14_test,prd14)
MSE14 = mean_squared_error(y14_test,prd14)
RMSE14 = np.sqrt(MSE14)
R2sc14 = r2_score(y14_test,prd14)
#print(plt.scatter(y14_test,prd14))
#plt.show()

# Life expectancy versus  thinness 5-9 years
X15 = df1[' thinness 5-9 years']
X15 = X15.values.reshape(-1,1)
sns.lmplot(data=df1,x=' thinness 5-9 years',y='Life expectancy ')
plt.show()
X15_train, X15_test, y15_train, y15_test = train_test_split(X15, y, test_size=0.3, random_state=99)
lm15 = LinearRegression()
lm15.fit(X15_train,y15_train) 
prd15 = lm15.predict(X15_test)
MAE15 = mean_absolute_error(y15_test,prd15)
MSE15 = mean_squared_error(y15_test,prd15)
RMSE15 = np.sqrt(MSE15)
R2sc15 = r2_score(y15_test,prd15)
#print(plt.scatter(y15_test,prd15))
#plt.show()

# Life expectancy versus Income composition of resources
X16 = df1['Income composition of resources']
X16 = X16.values.reshape(-1,1)
sns.lmplot(data=df1,x='Income composition of resources',y='Life expectancy ')
plt.show()
X16_train, X16_test, y16_train, y16_test = train_test_split(X16, y, test_size=0.3, random_state=99)
lm16 = LinearRegression()
lm16.fit(X16_train,y16_train) 
prd16 = lm16.predict(X16_test)
MAE16 = mean_absolute_error(y16_test,prd16)
MSE16 = mean_squared_error(y16_test,prd16)
RMSE16 = np.sqrt(MSE16)
R2sc16 = r2_score(y16_test,prd16)
#print(plt.scatter(y16_test,prd16))
#plt.show()

# Life expectancy versus Schooling
X17 = df1['Schooling']
X17 = X17.values.reshape(-1,1)
sns.lmplot(data=df1,x='Schooling',y='Life expectancy ')
plt.show()
X17_train, X17_test, y17_train, y17_test = train_test_split(X17, y, test_size=0.3, random_state=99)
lm17 = LinearRegression()
lm17.fit(X17_train,y17_train) 
prd17 = lm17.predict(X17_test)
MAE17 = mean_absolute_error(y17_test,prd17)
MSE17 = mean_squared_error(y17_test,prd17)
RMSE17 = np.sqrt(MSE17)
R2sc17 = r2_score(y17_test,prd17)
#print(plt.scatter(y17_test,prd17))
#plt.show()

coeffs = [lm1.coef_,lm2.coef_,lm3.coef_,lm4.coef_,lm5.coef_,lm6.coef_,lm7.coef_,
          lm8.coef_,lm9.coef_,lm10.coef_,lm11.coef_,lm12.coef_,lm13.coef_,
          lm14.coef_,lm15.coef_,lm16.coef_,lm17.coef_]

intercepts = [lm1.intercept_,lm2.intercept_,lm3.intercept_,lm4.intercept_,lm5.intercept_,
             lm6.intercept_,lm7.intercept_,lm8.intercept_,lm9.intercept_,lm10.intercept_,
             lm11.intercept_,lm12.intercept_,lm13.intercept_,lm14.intercept_,lm15.intercept_,
             lm16.intercept_,lm17.intercept_]

MAEs = [MAE1,MAE2,MAE3,MAE4,MAE5,MAE6,MAE7,MAE8,MAE9,MAE10,MAE11,MAE12,
        MAE13,MAE14,MAE15,MAE16,MAE17]

MSEs = [MSE1,MSE2,MSE3,MSE4,MSE5,MSE6,MSE7,MSE8,MSE9,MSE10,MSE11,MSE12,
        MSE13,MSE14,MSE15,MSE16,MSE17]

RMSEs = [RMSE1,RMSE2,RMSE3,RMSE4,RMSE5,RMSE6,RMSE7,RMSE8,RMSE9,RMSE10,
         RMSE11,RMSE12,RMSE13,RMSE14,RMSE15,RMSE16,RMSE17]

R2scores = [R2sc1,R2sc2,R2sc3,R2sc4,R2sc5,R2sc6,R2sc7,R2sc8,R2sc9,
            R2sc10,R2sc11,R2sc12,R2sc13,R2sc14,R2sc15,R2sc16,R2sc17]

Evals = pd.DataFrame(index=['intercepts','coeff','MAE','MSE','RMSE','R2score'],
                     data=[intercepts,coeffs,MAEs,MSEs,RMSEs,R2scores],
                     columns=predictors)

Evals1 = Evals.transpose()
print(Evals1)

Evals1.to_excel('Evals1prjct_slr.xlsx')

# Multiple linear regression model
Xm = df1[['Year', 'Adult Mortality','Alcohol', 'percentage expenditure', 
              'Hepatitis B', 'Measles ', ' BMI ','under-five deaths ', 'Polio', 'Total expenditure',
              'Diphtheria ',' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
              'Income composition of resources', 'Schooling']]
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, y, test_size=0.3, random_state=99)
lmm = LinearRegression()
lmm.fit(Xm_train,ym_train) 
prdm = lmm.predict(Xm_test)
MAEm = mean_absolute_error(ym_test,prdm)
MSEm = mean_squared_error(ym_test,prdm)
RMSEm = np.sqrt(MSEm)
R2scm = r2_score(ym_test,prdm)
print(plt.scatter(ym_test,prdm))
plt.show()
print('MLR intercept:',lmm.intercept_)
print('MLR coeffs:',lmm.coef_)
print('MLR MAE:',MAEm)
print('MLR MSE:',MSEm)
print('MLR RMSE:',RMSEm)
print('MLR R2score:',R2scm)
mlrcoeffs = lmm.coef_.reshape(-1,1)
cmdf = pd.DataFrame(data=mlrcoeffs,index=Xm.columns,columns=['Coeff'])
cmdf.reset_index(names='Predictors',inplace=True)
print(cmdf)

# prospatheia gia to p-value 
X2m = sm.add_constant(Xm)
est = sm.OLS(y, Xm)
est2 = est.fit()
print(est2.summary())
#prepei na symperilavw ta p-values gia na dw poies exoun shmantikes eksarthseis




