#Importing the required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics
import seaborn as sns
import statsmodels.api as sm

#Changing the current working directory
os.chdir('C:/Users/chait/Desktop/Project2')

#Reading the dataset
df = pd.read_csv('day.csv')

#Exploratory data analysis
df.dtypes

for i in range(0,9):
    
  df.iloc[:,i] = df.iloc[:,i].astype(object)
  
df.dtypes

#Checking for missing values
df.isnull().sum()
df.columns.values
list(df)

#Saving numeric columns
cnames = df.select_dtypes(exclude = ['object'])
numeric_variables = list(cnames.columns.values)
numeric_variables = numeric_variables[0:-1]

#Saving Categoical columns
cat_data = df.select_dtypes(include = ['object'])
categorical_variables = list(cat_data.columns.values)

#Plotting the variables using scatterplot

for i in numeric_variables :
    
   plt.scatter(df[i], df['cnt'], color='red')
   plt.title('Plot of cnt Vs '+i, fontsize=14)
   plt.xlabel(i, fontsize=14)
   plt.ylabel('cnt', fontsize=14)
   plt.grid(True)
   plt.show()

#Box plot 
plt.boxplot(df['temp'])
plt.boxplot(df['atemp'])
plt.boxplot(df['hum'])
plt.boxplot(df['windspeed'])
plt.boxplot(df['casual'])
plt.boxplot(df['registered'])

for i in numeric_variables:
    print(i)
    q75,q25 = np.percentile(df.loc[:,i],[75,25])
    iqr = q75-q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    df.loc[df.loc[:,i]<min ,i] = np.nan
    df.loc[df.loc[:,i]>max ,i] = np.nan     

#Checking the number of missing values aka outliers
df.isnull().sum()

#Replacing the outliers with median value    
df = df.fillna(df.median(),inplace = True)

#Feature Selection
f,ax = plt.subplots(figsize =(7,5))

#Generate correlation matrix
corr = cnames.drop(['cnt'],axis =1).corr()

#Plot using seaborne library
sns.heatmap(corr, mask =np.zeros_like(corr, dtype = np.bool), 
            cmap =sns.diverging_palette(220, 10, as_cmap =True),
            square = True, ax = ax)

#Feature selection
df = df.drop(['atemp'],axis =1)
df = df.drop(['instant'],axis =1)
df = df.drop(['dteday'],axis =1)

#Normalizing the data
a = df["casual"].max()
b = df["casual"].min()
df["casual"] = (df["casual"]- b)/(a-b)

a = df["registered"].max()
b = df["registered"].min()
df["registered"] = (df["registered"]- b)/(a-b)

#Dividing the dataset into Train and Test data
X = df.iloc[:,:-1]
Y = df.iloc[:,12]
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Building model using LinearRegression() method
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Predicting the Test set results

Y_Pred = regressor.predict(X_Test)

print ('Variance score: %.2f' % regressor.score(X_Test, Y_Pred))
print("R2 score : %.2f" % sklearn.metrics.r2_score(Y_Test,Y_Pred))



#Building regression model using statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y_Train, X_Train).fit()
predictions = model.predict(X_Test) 
 
print_model = model.summary()
print(print_model)

##############################################END##################################


