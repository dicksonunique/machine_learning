#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd

employee_data = pd.read_csv(r'C:\Users\Dickson\Downloads\Transform_Employee_dataset.csv', index_col=False)
employee_data.describe()
employee_data.drop(employee_data.index[employee_data['Normal']==989.900000], inplace= True) #dropping an outlier
employee_data.describe()


# In[10]:


# print(employee_data.to_string())
print(employee_data.columns)
employee_data.head()



# # from sklearn.tree import DecisionTreeRegressor
# 
# employee_data_clean = employee_data.fillna(employee_data.mean())
# features=['Normal', 'OT', 'Night Meal',
#        'Paid Leave', 'SOC COST','SERVICE CHARGE', 'VAT 7.5%']
# 
# employee_data_clean.rename('VAT 7.5%':'VAT')
# 

# In[58]:


# from sklearn.tree import DecisionTreeRegressor

employee_data_clean = employee_data.fillna(employee_data.mean())
features=['Normal', 'OT', 'Night Meal',
       'Paid Leave', 'SOC COST','SERVICE CHARGE', 'VAT']

employee_data_clean.rename(columns={'VAT 7.5%':'VAT'}, inplace=True)
employee_data_clean

# X.describe()


# In[94]:


import seaborn as sns
corr=employee_data_clean.corr(method='kendall')
corr
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)


# In[98]:


sns.pairplot(employee_data_clean)


# In[163]:


from sklearn.tree import export_graphviz,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
X=employee_data_clean[features]

y= employee_data_clean['TOTAL INVOICE']

X_train,X_test,y_train,y_test =train_test_split(X,y, random_state=1, test_size=0.26)

employee_data_model = DecisionTreeRegressor(random_state=0)

X.describe()
employee_data_model.fit(X_train,y_train)


# In[ ]:





# In[164]:


#making prediction
print("Making predictions for the following 5 Employees cost to the organisation:")
print(X.head())
print("The predictions are")
print(employee_data_model.predict(X_test.head()))


# In[165]:


#evalution
from sklearn.metrics import  accuracy_score,mean_squared_error,mean_absolute_error
X_predict = employee_data_model.predict(X_test)

print('Mean Absolute Error:',mean_absolute_error(y_test,X_predict))
print('Mean Squared Error:',mean_squared_error(y_test,X_predict))


# In[166]:


#The data of the actual and the predicted
df = pd. DataFrame({'Actual':y_test,'Predicted':X_predict})
df


# In[167]:


from sklearn import tree
import matplotlib.pyplot  as plt
plt.figure(figsize=(15,7))
tree.plot_tree(employee_data_model,filled=True,rounded=True,class_names=sorted(y.unique()),label='all',feature_names=X.columns)

