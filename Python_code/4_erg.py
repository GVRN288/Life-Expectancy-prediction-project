import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from scipy.cluster._hierarchy import linkage
from mpl_toolkits.mplot3d import Axes3D

# 4h ergasia

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

# Remove everything except Hepatitis B, Polio, Diptheria
df2.drop(['Adult Mortality', 'Alcohol', 'percentage expenditure',
                'Measles ', ' BMI ', 'under-five deaths ', 'Total expenditure',
                 ' HIV/AIDS', 'GDP', 'Population', ' thinness 5-9 years',
                 'Income composition of resources', 'Schooling', 'Life expectancy _True'], axis=1, inplace=True)

print(df2.info())


k3 = KMeans(n_clusters=3, random_state=101)
k3.fit(df2)
print('Cluster centers for k=3:', k3.cluster_centers_)
print('Cluster labels for k=3:', k3.labels_)

k4 = KMeans(n_clusters=4, random_state=101)
k4.fit(df2)
print('Cluster centers for k=4:', k4.cluster_centers_)
print('Cluster labels for k=4:', k4.labels_)

fig, axes = plt.subplots(1, 2, figsize=(11, 6))

axes[0].scatter(df2['Hepatitis B'], df2['Polio'], c=k3.labels_, cmap='viridis')
axes[0].set_title('K-means Clustering (k=3)')
axes[0].set_xlabel('Hepatitis B')
axes[0].set_ylabel('Polio')

axes[1].scatter(df2['Hepatitis B'], df2['Polio'], c=k4.labels_, cmap='viridis')
axes[1].set_title('K-means Clustering (k=4)')
axes[1].set_xlabel('Hepatitis B')
axes[1].set_ylabel('Polio')

plt.tight_layout()
plt.show()


# Elbow method to find the best number of K clusters

wcss = []
k_values = range(1,21)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=101)
    kmeans.fit(df2)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, 'bo-', markersize=8)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Determining Optimal k')
plt.grid(True)
plt.show()

# Hierarchical clustering with complete linkage
plt.figure(figsize=(10, 7))
plt.title('Dendrogram - Complete Linkage')
dg_complete = sch.dendrogram(sch.linkage(df2, method='complete'))
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.show()

# Hierarchical clustering with single linkage
plt.figure(figsize=(10, 7))
plt.title('Dendrogram - Single Linkage')
dg_single = sch.dendrogram(sch.linkage(df2, method='single'))
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.show()

