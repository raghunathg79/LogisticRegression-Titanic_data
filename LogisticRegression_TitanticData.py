#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics


# # PROBLEM STATEMENT - # WHAT TYPE OF PEOPLE LIKELY TO SURVIVE TITANIC TRAGEDY
# # CLUE - WOMAN, CHILDREN, CLASS 

# # Data Collection and Processing

# In[2]:


Train=pd.read_csv("train.csv")


# # Data Analysis

# In[3]:


Train.head()


# In[4]:


Train.tail()


# 1.Name, Sex, Cabin, Embarked are categorical variable
# 2.Survived, Pclass, Age, SibSp, Parch, Ticket, Fare are Numerica variables
# Action : Perform individual study for each variable to assess the data

# In[5]:


Train.describe()


# In[6]:


Train.info()


# In[7]:


Train.isnull().sum()


# # Null value treatment required "Age"

# In[8]:


Train.hist('Age', bins=[0,10,20,30,40,50,60,70,80,90,100])


# # We find 177 Null values for Age, Age being an important variable and % of null values morethan 5% we need to treat them accordingly.
# Considering the Histogram distribution of data across scale seems NORMAL we should apply "Mean"

# In[9]:


Train["Age"].fillna(Train["Age"].mean)


# In[10]:


Train["Age"].fillna(Train["Age"].mean(),inplace=True)


# In[11]:


Train


# # Categorical classification - Convert Male and Female into Integar form 

# In[12]:


Train.Sex[Train.Sex == 'male'] = 0
Train.Sex[Train.Sex == 'female'] = 1


# # create dummy variables for categorical features

# In[13]:


pclass_dummies = pd.get_dummies(Train.Pclass, prefix="Pclass")
embarked_dummies = pd.get_dummies(Train.Embarked, prefix="Embarked")


# # concatenate dummy columns with main dataset

# In[14]:


Train_dummies = pd.concat([Train, pclass_dummies, embarked_dummies], axis=1)
Train_dummies.drop(['Pclass','Embarked'], axis=1, inplace=True)


# In[15]:


Train_dummies.head()


# # Dropping headers which are not required

# In[16]:


Train = Train.drop(['Name','Ticket','Cabin'], axis=1)


# In[17]:


Train.head()


# In[18]:


Train.info()


# # Untill here we have applied EDA and ensured the Data is clean and complete for further analysis

# # Data Visualization

# In[19]:


sns.countplot('Survived', data=Train)
Train['Survived'].value_counts()


# In[20]:


sns.countplot('Sex', data=Train)
Train['Sex'].value_counts()


# During Categorical classification we have transformed categorical data to Numeric as part of EDA. Hence Male are classified as 0 and Female as 1

# In[21]:


sns.countplot('Sex', hue='Survived',data=Train)


# Survival rates higher in terms of Female (1) so during any such disaster the preferences of saving people categorically will be Females and Children.

# In[22]:


sns.countplot('Pclass', data=Train)


# In[23]:


sns.countplot('Pclass', hue='Survived',data=Train)


# Survival rates higher in terms of Class 1 passengers

# # Based on the Data Visualization charts we are able to determine Variables required to find out which Type of passengers survived Titanic Tragedy 

# # Create Model using Logistic regression - Testing Phase

# In[24]:


#Splitting the data to Test and Train the data by applying algorithms 


# In[25]:


X=Train[['Pclass','Age','Sex']]


# In[26]:


y=Train[['Survived']]


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0) 
# test size depicts the data is split into 80:20 ratio
# random_state signifies how the data sampling is done here its randomly picked
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[28]:


logreg=LogisticRegression()


# In[29]:


logreg.fit(X_train,y_train)


# # Model Evaluation

# In[30]:


y_predicted = logreg.predict(X_test)# Evaluation in terms of Test Data
y_predicted


# In[31]:


cnf_matrix = metrics.confusion_matrix(y_test,y_predicted)
cnf_matrix


# In[32]:


print("Accuracy:",metrics.accuracy_score(y_test, y_predicted))
print("Precision:",metrics.precision_score(y_test, y_predicted))
print("Recall:",metrics.recall_score(y_test, y_predicted))


# In[33]:


model = LogisticRegression()


# In[34]:


model.fit(X_train, y_train)


# In[35]:


X_train_prediction=model.predict(X_train)


# In[36]:


print(X_train_prediction)


# In[37]:


training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy score of training_data : ', training_data_accuracy)


# Conclusion :- Accuracy scoring for both Trained and Test data signifies precise 80% which means out of 100 data  points the Machine learning algorithm is able to predict 80 data points correctly
#     

# In[ ]:




