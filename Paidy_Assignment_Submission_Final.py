#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np


train=pd.read_csv('cs-training.csv')
test=pd.read_csv('cs-test.csv')


# In[82]:


## Check the columns with outliers or control risks to revise the train data for the modeling.
## The columns to be assessed are those 8 columns
#1.'RevolvingUtilizationOfUnsecuredLines'
#2. 'Age'
#3. 'DebtRatio'
#4. 'MonthlyIncome'
#5. 'NumberOfDependents'
#6. 'NumberOfTime30-59DaysPastDueNotWorse'
#7. 'NumberOfTime60-89DaysPastDueNotWorse'
#8. 'NumberOfTimes90DaysLate'


# In[83]:


## Assessment 1 - Revolving Utilization of Unsecured Lines

train['RevolvingUtilizationOfUnsecuredLines'].describe()


# In[84]:


test['RevolvingUtilizationOfUnsecuredLines'].describe()

## Both train & test data has Revolving Utilization of Unsecured Lines over 1.
## Though it implies the organization has credit limit control risk,
#  we need to use the data since the risk actually lies in the organization.


# In[85]:


## Assessment 2 - Age

train['age'].describe()


# In[86]:


test['age'].describe()

# The train data has an age value under 19, which is 0.
# The minimum age of test data is 21.
# Therefore, it is adequate to remove age = 0 line in the test data for the better tuned modeling.


# In[87]:


## Assessment 3 - Debt Ratio

train['DebtRatio'].describe()


# In[88]:


test['DebtRatio'].describe()

# Both train & test data has Revolving Utilization of Unsecured Lines over 1.
# Though it implies the organization has Debt to Income control risk,
#  we need to use the data since the risk actually lies in the organization.


# In[89]:


## Assessment 4 - Monthly Income-missing value

null_counts_MonthlyIncome_train = train.MonthlyIncome.isna().sum()
print(null_counts_MonthlyIncome_train)

null_counts_MonthlyIncome_test = test.MonthlyIncome.isna().sum()
print(null_counts_MonthlyIncome_test)


# In[90]:


# Remove rows with null values in 'MonthlyIncome' from train
train_dropna_MonthlyIncome = train.dropna(subset=['MonthlyIncome'])

# Calculate the median of the 'MonthlyIncome' column for train
median_monthly_income_train = train_dropna_MonthlyIncome['MonthlyIncome'].median()


# Remove rows with null values in 'MonthlyIncome' from test
test_dropna_MonthlyIncome = test.dropna(subset=['MonthlyIncome'])

# Calculate the median of the 'MonthlyIncome' column for test
median_monthly_income_test = test_dropna_MonthlyIncome['MonthlyIncome'].median()

print("Median Monthly Income for Train:", median_monthly_income_train)
print("Median Monthly Income for Test:", median_monthly_income_test)


# In[91]:


#Replace the missing value in the training data with 5400

train['MonthlyIncome'].fillna(5400, inplace=True)

# Calculate the median of the 'MonthlyIncome' column
median_monthly_income_train = train['MonthlyIncome'].median()

print("Median Monthly Income_train:", median_monthly_income_train)


# In[92]:


#Replace the missing value in the test data with 5400

test['MonthlyIncome'].fillna(5400, inplace=True)

# Calculate the median of the 'MonthlyIncome' column
median_monthly_income_test = test['MonthlyIncome'].median()

print("Median Monthly Income_test:", median_monthly_income_test)


# In[93]:


## Assessment 5 - NumberOfDependents-missing value

null_counts_NumberOfDependents_train = train.NumberOfDependents.isna().sum()
print(null_counts_NumberOfDependents_train)

null_counts_NumberOfDependents_test = test.NumberOfDependents.isna().sum()
print(null_counts_NumberOfDependents_test)


# In[94]:


# Remove rows with null values in 'NumberOfDependents' from train
train_dropna_dependents = train.dropna(subset=['NumberOfDependents'])

# Calculate the median of the 'NumberOfDependents' column for train
median_monthly_income_train = train_dropna_dependents['NumberOfDependents'].median()


# Remove rows with null values in 'NumberOfDependents' from test
test_dropna_dependents = test.dropna(subset=['NumberOfDependents'])

# Calculate the median of the 'NumberOfDependents' column for test
median_monthly_income_test = test_dropna_dependents['NumberOfDependents'].median()

print("Median Number Of Dependents for Train:", median_monthly_income_train)
print("Median Number O fDependents for Test:", median_monthly_income_test)


# In[95]:


#Replace the missing value in the training data with 0

train['NumberOfDependents'].fillna(0, inplace=True)

# Calculate the median of the 'NumberOfDependents' column
median_NumberOfDependents_train = train['NumberOfDependents'].median()

print("Median NumberOfDependents_train:", median_NumberOfDependents_train)


# In[96]:


#Replace the missing value in the test data with 0

test['NumberOfDependents'].fillna(0, inplace=True)

# Calculate the median of the 'NumberOfDependents' column
median_NumberOfDependents_test = test['NumberOfDependents'].median()

print("Median NumberOfDependents_test:", median_NumberOfDependents_test)


# In[97]:


## Assessment 6 - NumberOfTime30-59DaysPastDueNotWorse

train['NumberOfTime30-59DaysPastDueNotWorse'].describe()


# In[98]:


test['NumberOfTime30-59DaysPastDueNotWorse'].describe()

# Both have extra ordinary often past due history and implies the internal control risks.
# But we need to use the data since the risk actually lies in the organization.


# In[99]:


## Assessment 7 - NumberOfTime60-89DaysPastDueNotWorse

train['NumberOfTime60-89DaysPastDueNotWorse'].describe()


# In[100]:


test['NumberOfTime60-89DaysPastDueNotWorse'].describe()

# Both have extra ordinary often past due history and implies the internal control risks.
# But we need to use the data since the risk actually lies in the organization.


# In[101]:


## Assessment 8 - NumberOfTimes90DaysLate
train['NumberOfTimes90DaysLate'].describe()


# In[102]:


test['NumberOfTimes90DaysLate'].describe()

# Both have extra ordinary often past due history and implies the internal control risks.
# But we need to use the data since the risk actually lies in the organization.


# In[103]:


## Remove Age = 0 line in the train data before the modeling.

train_Final = train[train['age'] != 0]
train_Final['age'].describe()


# In[104]:


## Replacing Unnamed column name with "Identifier" and removing in the train data before the modeling.

# Replace 'Unnamed: 0' with the actual column name you want to replace and remove
column_to_replace = 'Unnamed: 0'

# Rename the column to 'Identifier'
train_Final = train_Final.rename(columns={column_to_replace: 'Identifier'})

# Drop the 'Identifier' column
train_Final = train_Final.drop(columns=['Identifier'])


# In[105]:


## Replacing Unnamed column name with "Identifier" and removing in the test data before the modeling.

# Replace 'Unnamed: 0' with the actual column name you want to replace and remove
column_to_replace_test = 'Unnamed: 0'

# Rename the column to 'Identifier'
test = test.rename(columns={column_to_replace_test: 'Identifier'})

# Drop the 'Identifier' column
test = test.drop(columns=['Identifier'])


# In[106]:


train_Final.describe()


# In[107]:


##### Create Random Forest Model #####

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Define features and target
features = train_Final.drop('SeriousDlqin2yrs', axis=1)
target = train_Final['SeriousDlqin2yrs']


# In[108]:


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)


# In[109]:


# Create and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)


# In[110]:


# Evaluate the model on the validation set
val_predictions = rf_classifier.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy)


# In[111]:


# Use the trained model to predict the 'SeriousDlqin2yrs' column in the test data
test_features = test.drop('SeriousDlqin2yrs', axis=1)
test_predictions = rf_classifier.predict(test_features)


# In[112]:


# Add the predictions as a new column in the test data
test['Predicted_SeriousDlqin2yrs'] = test_predictions


# In[113]:


# Save the test data with predictions to a CSV file
test.to_csv('cs-test-predictions.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




