#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd

employee_data = pd.read_csv(r'C:\Users\Dickson\Downloads\Transform_Employee_dataset.csv', index_col=False)
employee_data.describe()
employee_data.drop(employee_data.index[employee_data['Normal']==989.900000], inplace= True) #dropping an outlier
employee_data.describe()


# In[93]:


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

# In[94]:


# from sklearn.tree import DecisionTreeRegressor

employee_data_clean = employee_data.fillna(employee_data.mean())
features=['Normal', 'OT', 'Night Meal',
       'Paid Leave', 'SOC COST','SERVICE CHARGE', 'VAT']

employee_data_clean.rename(columns={'VAT 7.5%':'VAT'}, inplace=True)
employee_data_clean

X.describe()


# In[97]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
X=employee_data_clean[features]

y= employee_data_clean['TOTAL INVOICE']

employee_data_model = DecisionTreeRegressor(random_state=1)

X.describe()
employee_data_model.fit(X,y)


# In[100]:


#making prediction
print("Making predictions for the following 5 Employees costto the organisation:")
print(X.head())
print("The predictions are")
print(employee_data_model.predict(X.head()))

