#Clear workspace
rm(list = ls())

#Set the working directory
setwd('C:/Users/chait/Desktop/Project2')

#Load Libraries
x = c('ggplot2', 'corrgram', 'DMwR', 'caret', 'randomForest', 'unbalanced', 'C50', 'dummies', 
      'e1071', 'Information','MASS','rpart', 'gbm', 'ROSE') 

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#Read the dataset
df = read.csv('day.csv')

#Data preprocessing
sum(is.na(df))
head(df)
str(df)
summary(df)
colnames(df)

df[,1:9] =as.data.frame(lapply(df[,1:9],as.factor))


#Seperating the numeric data
numeric_index = sapply(df,is.numeric)
numeric_data = df[,numeric_index]

cnames = colnames(numeric_data)

cnames = cnames[1:6] #Removing the dependent variable 

#Exploratory data analysis
for(i in 1:ncol(numeric_data))
 {
  hist(numeric_data[,i],main = paste("Histogram of" , cnames[i]),col='green',border = 'black',xlab = cnames[i])
 }

#Scatterplot of numeric variables Vs Dependent variable
#Exploratory data analysis
for(i in 1:length(cnames))
 {
  x = df[,cnames[i]]
  plot(x , y = df$cnt, main = paste("cnt vs ", cnames[i]),col = "blue")
 }    

#Smooth Scatter plot describing the relation ship between dependent and numeric variables

for(i in 1:(length(cnames)))
 {
  scatter.smooth(x=df[,cnames[i]],y=df$cnt,xlab = cnames[i],ylab = "cnt",col = "blue", main=paste("Smooth Scatter plot of cnt Vs ",cnames[i])) 
 }

#Outlier Analysis
#Boxplot of numeirc variables
for(i in 1:(length(cnames)))
 {
  boxplot(df[,cnames[i]],main = paste("Box Plot of ",cnames[i]),col = "green", outlier.color = "red", outcol ="red",sub = paste("Outliers: ",length(boxplot.stats(df[,cnames[i]])$out)))
 }


for( i in cnames)
  {
   print(i)
   val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
   df[,i][df[,i] %in% val] = NA
  }
sum(is.na(df))

#Imputing the missing values with knnImputation
df = knnImputation(df, k=80)

#Feature Selection
library(corrplot)
library(corrgram)
M= cor(numeric_data[,-7])
corrplot(M,method = "circle")
corrplot(M, type="upper")

#VIF
library(usdm)
#vif(df[,-16])
vifcor(numeric_data[,-10],th = 0.9)

df1 = subset(df, select = -c(atemp,instant,dteday))

#Feature scaling 

df[,i] = (df[,"casual"]- min(df[,"casual"]))/
  (max(df[,"casual"] - min(df[,"casual"])))   
df[,i] = (df[,"registered"]- min(df[,"registered"]))/
  (max(df[,"registered"] - min(df[,"registered"])))   


#Dividing Test and train datasets
train_index <- sample(1:nrow(df1), 0.8 * nrow(df1))
test_index <- setdiff(1:nrow(df1), train_index)

# Build X_train, y_train, X_test, y_test
X_train <- df1[train_index, -13]
y_train <- df1[train_index, "cnt"]

X_test <- df1[test_index, -13]
y_test <- df1[test_index, "cnt"]

df_train = data.frame(X_train,y_train)
colnames(df_train)[13] = "cnt"

#Linear Model
LinearModel = lm(cnt~., data = df_train)
summary(LinearModel)
confint(LinearModel)

Y_pred = predict(LinearModel,newdata = X_test)
predict(LinearModel,type = "terms")

