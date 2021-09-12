#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[3]:


dataframe=pd.read_csv('C://Users//b.i.s//Desktop//student_score.csv')
dataframe


# In[4]:


#Next is to view the data and so we are using the head() function.The head function default view is the top five, but again
#whatever you want to be in the number of views you can do it as entered.
dataframe.head()


# In[5]:


#Next is to view using describe() function which is used to view some basic statistical details like percentile, mean, std etc
dataframe.describe()


# In[6]:


#Next is view The info() function is used to print a concise summary of a DataFrame. This method prints information about a
#DataFrame including the index dtype and column dtypes,non-null values and memory usage. Whether to print the full summary. 
#By default, the setting in pandas
dataframe.info()


# In[7]:


#Next view the Histogram `of student study hours and student marks or score using hist() function
dataframe.hist()


# In[8]:


#The next phase is to enter distribution scores and plot them according to the requirement, here we are going to enter the title,
#x_label, and y_label, and show it according to the desired result.
dataframe.plot(x='Study_hrs', y='Student_marks', style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('Study_hrs')    
plt.ylabel('Student_marks')    
plt.show() 


# Prepare the data for machine learning algorthim

# In[9]:


#Data Cleasing
dataframe.isnull().sum()


# In[10]:


dataframe.mean()


# In[11]:


data=dataframe.fillna(dataframe.mean())


# In[12]:


data.isnull().sum()


# In[13]:


data.head()


# #Split Dataset for training and testing

# In[17]:


#The split of data into the training and test sets is very important as in this time we will be using 
#Scikit Learn's builtin method of train_test_split()
# Defining X and y from the Data
X = dataframe.iloc[:, :-1].values  
y = dataframe.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# Fitting the data into model

# In[20]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# Predicting the percentage of marks
# 

# In[21]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# Compare the predicted marks with actual marks

# In[22]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[23]:


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[24]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# 
# Small value of Mean absolute error states that the chances of error or wrong forecasting through the model are very less.
# 
# What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?
# 

# In[25]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# 
# According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.

# In[ ]:





# In[ ]:




