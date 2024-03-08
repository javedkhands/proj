#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[3]:


df=pd.read_csv("C:/Users/bhaim/Downloads/advertising.csv")


# In[4]:


df


# In[6]:


""" Data Description

- TV: This column represents the amount of money spent on advertising through TV channels.
    
    It indicates the financial investment in television advertising for each instance.

- Radio: This column represents the advertising expenditure on radio. 
    It shows how much money is allocated to radio advertising for each case.

- Newspaper: This column represents the advertising expenditure in newspapers.
    It indicates the financial investment in newspaper advertising for each instance.

- Sales: This column represents the sales generated as a result of the advertising expenditures in TV, Radio, and Newspaper.
It shows the outcome variable that is being analyzed or predicted.

The dataset contains information about advertising expenditures across different channels (TV, Radio, Newspaper) and
the corresponding sales outcomes. The goal is to analyze the relationship between advertising spending and sales, 
potentially building a predictive model or understanding which advertising channels contribute more to sales."""


# In[7]:


print(f"The shape of the data is: {df.shape}")
print(f"The size of the data is: {df.size}")


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


if df.duplicated().sum() > 0:
    print(f"There are {df.duplicated().sum()} duplicated rows in the dataset.")
else:
    print("There are no duplicated rows in the dataset.")


# In[11]:


if df.isnull().sum().any():
    print(f"There are {df.isnull().sum().sum()} null values in the dataset.")
else:
    print("There are no null values in the dataset.")


# In[12]:


plt.figure(figsize=(8, 6))

sns.boxplot(x=df["TV"], color='skyblue', width=0.3, linewidth=2)

plt.title("Distribution of TV Feature", fontsize=16)
plt.xlabel("TV", fontsize=14)
plt.ylabel("Values", fontsize=14)

plt.show()


# In[13]:


x_all=df.iloc[:,0:3]
x=df.iloc[:,[0]]
y=df.iloc[:,-1]


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train.size,x_test.size


# In[15]:


"""Linear regression, a statistical method for modeling the relationship between a dependent variable and one or
more independent variables, relies on several key assumptions. 
These assumptions are crucial for ensuring the validity and reliability of the model's results."""

from sklearn.linear_model import LinearRegression
x_train_reshaped = x_train.values
y_train_reshaped = y_train.values 
model = LinearRegression()
model.fit(x_train_reshaped, y_train_reshaped)


# In[17]:


"""This assumption states that the relationship between the dependent variable and the independent variables is linear. 
In simpler terms, the change in the dependent variable should be proportional to the change in the independent variable(s). 
This can be visualized as a straight line in a scatter plot of the data."""

x_test_reshaped=x_test.values.reshape(-1,1)
y_pred=model.predict(x_test_reshaped)
residual=y_test-y_pred
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales',
                               height=3,size=4, aspect=1, kind='scatter', markers='o')


# In[18]:


"""# 2) Multicollinearity
This assumption states that the independent variables are not highly correlated with each other.
Multicollinearity occurs when there is a strong linear relationship between two or more independent variables. This can lead to 
unreliable estimates of the coefficients and make it difficult to interpret the results of the regression analysis."""

"""Correlation Heatmap
The correlation heatmap is a useful tool for assessing multicollinearity among predictor variables in a dataset.
In a correlation heatmap:
Darker colors (towards -1 or 1) indicate stronger correlations.
Positive values indicate a positive correlation, while negative values indicate a negative correlation."""

sns.heatmap(df.iloc[:,0:3].corr(),annot=True)


# In[19]:


"""Normal Residual¶
The residuals (the differences between the observed and predicted values) are assumed to be normally distributed."""
sns.displot(residual,kind="kde",fill=True)
plt.title("Kernel Density Estimate")
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.show()


# In[20]:


# QQ Plot
import scipy as sp

fig,ax =plt.subplots(figsize=(6,4))
sp.stats.probplot(residual,plot=ax,fit=True)

plt.show()

"""Based on the conducted analysis, it is apparent that the residuals demonstrate a normal distribution. This observation 
is supported by the examination of both Quantile-Quantile (QQ) plots and Kernel Density Estimation (KDE) plots."""


# In[21]:


"""No Autocorrelation of Error¶
This assumption states that the error terms (the difference between the actual values and the predicted values) 
are not correlated with each other.
In simpler terms, the error at one observation should not be related to the error at any other observation."""
sns.lineplot(x=range(len(residual)), y=residual)

plt.xlabel("Index")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()

"""The absence of repetitive patterns observed in the analysis indicates that the residuals are not autocorrelated.
This observation is further validated by utilizing the sns.relplot with
the kind parameter set to "line" to visually inspect and confirm the absence of autocorrelation in the residuals."""


# In[22]:


# Interpretation of Linear Regression Results


# In[23]:


y_pred


# In[24]:


print(f"The slope of the best fit is {model.coef_}")
print(f"The intercept of the best fit is {model.intercept_}")


# In[25]:


sns.regplot(x=x_train_reshaped, y=y_train_reshaped, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title("Regression Plot")
plt.xlabel("Predictor (x)")
plt.ylabel("Target Variable (y)")
plt.show()


# In[26]:


"""Regression Metrics

Commonly used regression performance metrics in scikit-learn include:

Mean Absolute Error (MAE): Measures the average absolute differences between predicted and actual values.
Mean Squared Error (MSE): Measures the average squared differences between predicted and actual values.
Root Mean Squared Error (RMSE): The square root of the MSE; provides an interpretable scale.
These metrics collectively provide a comprehensive picture of how well a regression model is
fitting the data, and they serve as valuable tools in the model development and evaluation process."""


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

print(f"The Mean Absolute Error obtained is : {mae}")
print(f"The Mean Squared Error obtained is : {mse}")
print(f"The Root Mean Squared Error obtained is:{rmse}")


# In[ ]:




