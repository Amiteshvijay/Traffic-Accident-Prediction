#!/usr/bin/env python
# coding: utf-8

# ## 1. Problem Understanding and Goal Definition
# 1. Define the problem: 
#     Predict the likelihood of a traffic accident based on factors like Weather, Road_Type, Time_of_Day, Traffic_Density, Speed_Limit,Number_of_Vehicles, Driver_Alcohol, Accident_Severity,Road_Condition, Vehicle_Type, Driver_Age, Driver_Experience,Road_Light_Condition, Accident etc.
# 2. Key Questions:
#     1.  What data is needed? (e.g., historical accident records, traffic data, weather conditions).
#     2.  What is the prediction goal? (Binary classification: accident/no accident, or regression: accident severity).
#     3. Who will use the model? (Traffic authorities, logistics companies, etc.).

# Notes: This dataset can be used to train classification models to predict whether an accident will occur based on these factors. You can apply machine learning algorithms such as logistic regression, random forests, gradient boosting, or neural networks to build a predictive model.

# In[63]:


##import necessaries libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('dataset_traffic_accident_prediction1.csv')


# In[3]:


df.head()


# ### Exploratory Data Analysis (EDA):
#         1. It is the process of analyzing and visualizing datasets to uncover patterns, relationships, and insights. 
# #### For a Traffic Accident Prediction project, EDA is critical to understand the dataset, identify anomalies, and create meaningful features.

# ### 1. Data Overview
# ##### Objective: Understand the structure and content of the dataset.

# In[4]:


#To know the rows and columns in the dataset
df.shape


# In[5]:


#Feature names that will help us to predict the models
df.columns


# In[6]:


# Identyifies the information regarding data like feature's name, Non-Null value count and also about data types.
df.info()


# In[7]:


# get statisticsal summary for numerical columns in the dataset
df.describe()


# In[8]:


# Get sum of missing values in the dataset
df.isnull().sum()


# ### 2. Handling Missing Data
#  1. Objective: Address missing or incomplete data.
#  2. Data can be two types 
#      1. Categorical Data Types
#      2. Numerical Data Types

# ##### Handle missing value based on categorical data types

# In[9]:


df.info()


# In[10]:


df.head()


# In[11]:


df.columns


# ###### categorical data types
# 1. 'Weather', 'Road_Type', 'Time_of_Day','Accident_Severity','Road_Condition', 'Vehicle_Type','Road_Light_Condition'
# ###### Numerical data types
# 1. 'Traffic_Density', 'Speed_Limit','Number_of_Vehicles', 'Driver_Alcohol','Driver_Age', 'Driver_Experience','Accident'

# In[12]:


training_dataset=df[df['Accident'].isnull()]


# In[13]:


df['Road_Type'].isnull().sum()


# In[14]:


df['Weather'].unique()


# In[15]:


df['Weather']=df['Weather'].fillna(df['Weather'].mode()[0])
df['Road_Type']=df['Road_Type'].fillna(df['Road_Type'].mode()[0])
df['Time_of_Day']=df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
df['Accident_Severity']=df['Accident_Severity'].fillna(df['Accident_Severity'].mode()[0])
df['Vehicle_Type']=df['Vehicle_Type'].fillna(df['Vehicle_Type'].mode()[0])
df['Road_Light_Condition']=df['Road_Light_Condition'].fillna(df['Road_Light_Condition'].mode()[0])


# In[16]:


df.isnull().sum()


# ###### Numerical data types
# 

# In[17]:


df.columns


# In[18]:


sns.histplot(df['Traffic_Density'],kde=True)


# #### Check for Outliers:
# 
# 1. Use techniques like the interquartile range (IQR) to detect outliers.

# In[19]:


Q1 = df['Traffic_Density'].quantile(0.25)
Q3 = df['Traffic_Density'].quantile(0.75)
IQR = Q3 - Q1
print("Outliers:", ((df['Traffic_Density'] < (Q1 - 1.5 * IQR)) | (df['Traffic_Density'] > (Q3 + 1.5 * IQR))).sum())


# In[20]:


df['Traffic_Density'] = df['Traffic_Density'].fillna(df['Traffic_Density'].mean())


# In[21]:


df.info()


# In[22]:


Q1 = df['Speed_Limit'].quantile(0.25)
Q3 = df['Speed_Limit'].quantile(0.75)
IQR = Q3 - Q1
print("Outliers:", ((df['Speed_Limit'] < (Q1 - 1.5 * IQR)) | (df['Speed_Limit'] > (Q3 + 1.5 * IQR))).sum())


# In[23]:


sns.histplot(df['Speed_Limit'], kde=True)


# In[24]:


df['Speed_Limit'] = df['Speed_Limit'].fillna(df['Speed_Limit'].median())
df['Number_of_Vehicles'] = df['Number_of_Vehicles'].fillna(df['Number_of_Vehicles'].mode()[0])


# In[25]:


df['Number_of_Vehicles'].unique()


# In[26]:


df['Number_of_Vehicles'].median()


# In[27]:


df['Number_of_Vehicles'].mode()[0]


# In[28]:


Q1 = df['Number_of_Vehicles'].quantile(0.25)
Q3 = df['Number_of_Vehicles'].quantile(0.75)
IQR = Q3 - Q1
print("Outliers:", ((df['Number_of_Vehicles'] < (Q1 - 1.5 * IQR)) | (df['Number_of_Vehicles'] > (Q3 + 1.5 * IQR))).sum())


# In[29]:


sns.histplot(df['Number_of_Vehicles'], kde=True)


# In[ ]:





# In[30]:


#numerical categorical variables, don't use mean/median, we use mode

df['Driver_Alcohol'] = df['Driver_Alcohol'].fillna(df['Driver_Alcohol'].mode()[0])


# In[31]:


sns.histplot(df['Driver_Alcohol'],kde=True)


# In[32]:


df['Driver_Age'].isnull().sum()


# In[33]:


df['Driver_Age'].median()


# In[34]:


df['Driver_Age'].mean()


# In[35]:


Q1 = df['Driver_Age'].quantile(0.25)
Q3 = df['Driver_Age'].quantile(0.75)
IQR = Q3 - Q1
print("Outliers:", ((df['Driver_Age'] < (Q1 - 1.5 * IQR)) | (df['Driver_Age'] > (Q3 + 1.5 * IQR))).sum())


# In[ ]:





# In[36]:


Q1 = df['Driver_Experience'].quantile(0.25)
Q3 = df['Driver_Experience'].quantile(0.75)
IQR = Q3 - Q1
print("Outliers:", ((df['Driver_Experience'] < (Q1 - 1.5 * IQR)) | (df['Driver_Experience'] > (Q3 + 1.5 * IQR))).sum())


# In[37]:


df['Driver_Age'] = df['Driver_Age'].fillna(df['Driver_Age'].mean())
df['Driver_Experience'] = df['Driver_Experience'].fillna(df['Driver_Experience'].mean())


# In[38]:


df.isnull().sum()


# In[39]:


df['Road_Condition'].unique()


# In[40]:


df['Road_Condition'] = df['Road_Condition'].fillna(df['Road_Condition'].mode()[0])


# In[53]:


# Drop rows where 'column_name' has NaN values
df = df.dropna(subset=['Accident'])


# In[41]:


df.isnull().sum()


# In[54]:





# In[55]:


df.info()


# In[59]:


#Machine learning models require numerical inputs, so categorical variables need to be encoded
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Weather'] = encoder.fit_transform(df['Weather'])
df['Road_Type'] = encoder.fit_transform(df['Road_Type'])
df['Time_of_Day'] = encoder.fit_transform(df['Time_of_Day'])
df['Accident_Severity'] = encoder.fit_transform(df['Accident_Severity'])
df['Road_Condition'] = encoder.fit_transform(df['Road_Condition'])
df['Vehicle_Type'] = encoder.fit_transform(df['Vehicle_Type'])
df['Road_Light_Condition'] = encoder.fit_transform(df['Road_Light_Condition'])


# In[60]:


df.info()


# In[61]:


# # Normalize Traffic_Volume
# scaler = MinMaxScaler()
# df_encoded['Normalized_Volume'] = scaler.fit_transform(df_encoded[['Traffic_Volume']])

# # Standardize Speed_Limit
# scaler = StandardScaler()
# df_encoded['Standardized_Speed'] = scaler.fit_transform(df_encoded[['Speed_Limit']])


# #### Exploratory Data Analysis (EDA)

# In[66]:


# Set figure size and heatmap grid size
plt.figure(figsize=(15, 10))  # Set figure size (width, height)

corr = df.corr()  # Compute correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.8)
plt.title('Correlation Heatmap')
plt.show()


# ### Visualize Relationships
# 1. Correlate Accidents with Weather

# In[69]:


# df should contain columns like ['Weather', 'Accident_Severity', 'Accident_Count']
sns.countplot(data=df, x='Weather', hue='Accident_Severity')
plt.title('Accidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()


# #### Hypothesize: Rainy Weather Increases Accident Severity

# In[70]:


severity_weather = df.groupby(['Weather', 'Accident_Severity']).size().unstack()
severity_weather.plot(kind='bar', figsize=(10, 6))
plt.title('Accident Severity by Weather')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.legend(title='Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[71]:


sns.pairplot(df[['Traffic_Density', 'Speed_Limit', 'Accident_Severity']], hue='Accident_Severity')
plt.show()


# In[72]:


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = df.corr()

# Visualize the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Drop features with low correlation with the target
high_corr_features = corr_matrix['Accident_Severity'][abs(corr_matrix['Accident_Severity']) > 0.1]
print("Highly Correlated Features:\n", high_corr_features)


# In[73]:


high_corr_features


# ### Statistical Tests :Use statistical methods to determine feature relevance:
# 
# 1. For Numerical Features:
# 
# ANOVA (Analysis of Variance): Test the variance between the target and numerical variables.
# 
# Pearson’s Correlation: Measures the linear relationship between variables.
# 
# 2. For Categorical Features:
# 
# Chi-Square Test: Assesses the relationship between categorical variables and the target.

# In[74]:


from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
encoder = LabelEncoder()
df['Road_Type_Encoded'] = encoder.fit_transform(df['Road_Type'])

# Perform Chi-Square Test
chi_scores, p_values = chi2(df[['Road_Type_Encoded']], df['Accident_Severity'])
print("Chi-Square Scores:", chi_scores)
print("P-Values:", p_values)


# In[78]:


from sklearn.model_selection import train_test_split

# Split the dataset
X = df.drop(columns=['Accident','Road_Type_Encoded'])  # Features
y = df['Accident']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Check the class distribution
print(y_train.value_counts(normalize=True))  # Proportions of each class in the train set


# In[80]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[81]:


from sklearn.model_selection import GridSearchCV

    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
# Define hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Apply Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1_weighted',

# Best parameters
print("Best Parameters:", grid_search.best_params_)


# In[83]:


from sklearn.metrics import confusion_matrix, roc_auc_score

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# # ROC-AUC Score (if applicable)
# roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')
# print("ROC-AUC Score:", roc_auc)


# #### Train the model with Logistics regression 

# In[87]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42, class_weight='balanced')
# Train a Random Forest model
log_reg.fit(X_train,y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[ ]:




