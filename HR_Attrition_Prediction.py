#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[50]:


employee_df = pd.read_csv('D:\HR project\Human_Resources.csv')


# In[51]:


employee_df


# In[52]:


pd.set_option('display.max_columns', None)


# In[53]:


employee_df.head()


# In[54]:


employee_df.describe()


# In[55]:


employee_df.info()


# In[56]:


# we convert binary categorical columns such as `Yes/No` and `Y/N` into numerical values (`1` and `0`).

employee_df['Attrition']=employee_df['Attrition'].apply (lambda x:1 if x=='Yes' else 0)
employee_df['Over18']=employee_df['Over18'].apply (lambda x:1 if x=='Y'  else 0 )
employee_df['OverTime']=employee_df['OverTime'].apply(lambda x:1 if x=='Yes' else 0)


# This helps in:
# - Performing statistical analysis and correlation calculations
# - Making visualizations easier
# - Preparing the dataset for machine learning models
# 

# In[10]:


employee_df


# In[57]:


print(employee_df['Attrition'].value_counts())


# In[58]:


sns.heatmap(employee_df.isnull (), yticklabels = False, cbar = False, cmap="Blues")


# -There is no null values

# In[59]:


employee_df.hist(bins=30,figsize = (20,20), color = 'b')
employee_df.hist(bins=30,figsize = (20,20), color = 'b')


# -Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tail heavy

# In[60]:


for col in ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']:
    print(col)
    print(employee_df[col].unique())
    print('----------------')


# -They are constant columns and do not provide useful information for analysis or machine learning models because they have no variation and EmployeeNumber is simply an identifier column and does not contribute to prediction.
# 

# -Therefore, these columns can be safely dropped from the dataset.

# In[61]:


employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'] , axis=1 , inplace=True)


# In[62]:


left=employee_df[employee_df['Attrition']==1]
stayed=employee_df[employee_df['Attrition']==0]


# -Count the number of employees who stayed and left

# In[63]:


print ('Total=', len(employee_df))
print("Number of employees who left the company =", len(left))
print("Number of employees who did not leave the company (stayed) =", len(stayed))


# -It seems that we are dealing with an imbalanced dataset 

# In[64]:


stayed.describe()


# In[65]:


left.describe()


#  1.'age': mean age of the employees who stayed is higher compared to who left
# 
# 
# 
# 
# 

# 2. 'DailyRate': Rate of employees who stayed is higher

# 3. 'DistanceFromHome': Employees who stayed live closer to home 

# 4. 'EnvironmentSatisfaction' & 'JobSatisfaction': Employees who stayed are generally more satisifed with their jobs

#   5.'StockOptionLevel': Employees who stayed tend to have higher stock option level

# In[66]:


correlations = employee_df.corr(numeric_only=True)
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)


# -Job level is strongly correlated with total working hours

# -Monthly income is strongly correlated with Job level

# -Monthly income is strongly correlated with total working hours

# -Age is stongly correlated with monthly income

# In[67]:


plt.figure(figsize=[25,12])
sns.countplot(x='Age' , hue='Attrition', data=employee_df)


# In[68]:


plt.figure(figsize=[20,20])
plt.subplot(411)
sns.countplot(x='JobRole', hue = 'Attrition', data = employee_df)
plt.subplot(412)
sns.countplot(x='MaritalStatus', hue = 'Attrition', data = employee_df)
plt.subplot(413)
sns.countplot(x='JobInvolvement', hue = 'Attrition', data = employee_df)
plt.subplot(414)
sns.countplot(x='JobLevel', hue = 'Attrition', data = employee_df)


# -Single employees tend to leave compared to married and divorced

# -Sales Representitives tend to leave compared to any other job

# -Less involved employees tend to leave the company

# -Less experienced (low job level) tend to leave the company

# In[69]:


plt.figure(figsize=(12,7))

sns.kdeplot(left['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed['DistanceFromHome'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Distance From Home')


# - Most employees live relatively close to work.

# - Employees with longer commuting distances appear more likely to leave.

# - Attrition density is slightly higher for employees living farther away.

# Long commuting distances may contribute to employee dissatisfaction and increase the probability of attrition.
# Work-life balance and commuting convenience can impact employee retention.

# In[70]:


plt.figure(figsize=(12,7))

sns.kdeplot(left['YearsWithCurrManager'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed['YearsWithCurrManager'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Years With Current Manager')


# - Most employees stayed with their manager between 0–4 years.

# - Employees with very short manager relationships show higher attrition.

# - Attrition decreases as years with the current manager increase.

# - Long-term manager relationships are associated with employee retention.

# Managerial stability may improve employee satisfaction and retention.

# In[71]:


plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'Gender', data = employee_df)


# -There is no significant difference between the average salaries of males and females in the boxplot.

# In[72]:


plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df)


# - Managers and Research Directors have the highest salaries.
# 

# - Sales Representatives and Laboratory Technicians have lower salaries.

# -Salary strongly depends on the employee's job role and position level.

# In[ ]:





# # Data preprocessing

# In[73]:


employee_df.head(3)


# -It seems like we have some categorical columns should be convert

# In[85]:


x_cat = ['BusinessTravel', 'Department', 'EducationField',
         'Gender', 'JobRole', 'MaritalStatus']


# In[86]:


x_numerical = ['Age', 'DailyRate', 'DistanceFromHome',
               'Education', 'EnvironmentSatisfaction',
               'HourlyRate', 'JobInvolvement', 'JobLevel',
               'JobSatisfaction', 'MonthlyIncome',
               'MonthlyRate', 'NumCompaniesWorked',
               'PercentSalaryHike', 'PerformanceRating',
               'RelationshipSatisfaction',
               'StockOptionLevel', 'TotalWorkingYears',
               'TrainingTimesLastYear',
               'WorkLifeBalance', 'YearsAtCompany',
               'YearsInCurrentRole',
               'YearsSinceLastPromotion',
               'YearsWithCurrManager','OverTime']


# In[87]:


print(employee_df['Attrition'].value_counts())


# In[88]:


X = employee_df[x_cat + x_numerical]
y = employee_df['Attrition']


# In[89]:


print(employee_df['Attrition'].value_counts())


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[91]:


print(employee_df['Attrition'].value_counts())


# In[92]:


print(y_train.value_counts())
print(y_test.value_counts())


# In[93]:


categorical_transformer = OneHotEncoder(drop='first')


# In[94]:


numerical_transformer = MinMaxScaler()


# In[95]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, x_cat),
        ('num', numerical_transformer, x_numerical)
    ]
)


# In[96]:


X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# In[47]:


model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)


# In[97]:


model.fit(X_train_processed, y_train)


# In[98]:


y_pred = model.predict(X_test_processed)


# In[99]:


print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[101]:


rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)


# In[103]:


rf_model.fit(X_train_processed, y_train)


# In[104]:


y_pred_rf = rf_model.predict(X_test_processed)


# In[105]:


print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# The model is heavily biased toward class 0 (employees staying).
# 
# It predicts almost everyone as staying.
# 
# That’s why:
# 
# accuracy is high
# but attrition detection is poor

# In[ ]:





# In[108]:


X_train_processed.shape


# In[110]:


ann_model = Sequential()


# In[113]:


ann_model = Sequential()

ann_model.add(Dense(
    units=64,
    activation='relu',
    input_dim=44
))


# In[114]:


ann_model.add(Dropout(0.3))


# In[115]:


ann_model.add(Dense(
    units=32,
    activation='relu'
))


# In[116]:


ann_model.add(Dropout(0.3))


# In[117]:


ann_model.add(Dense(
    units=1,
    activation='sigmoid'
))


# In[118]:


ann_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[119]:


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# In[120]:


history = ann_model.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    class_weight={0:1, 1:3},
    callbacks=[early_stop]
)


# In[121]:


y_pred_prob = ann_model.predict(X_test_processed)


# In[122]:


y_pred_ann = (y_pred_prob > 0.5).astype(int)


# In[123]:


print("Accuracy:", accuracy_score(y_test, y_pred_ann))
print(confusion_matrix(y_test, y_pred_ann))
print(classification_report(y_test, y_pred_ann))


# In[ ]:




