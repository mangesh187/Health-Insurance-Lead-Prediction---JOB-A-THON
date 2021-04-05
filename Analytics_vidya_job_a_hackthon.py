#!/usr/bin/env python
# coding: utf-8

# # Health Insurance Lead Prediction - JOB-A-THON

# # Step 1: Reading and Understanding the Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use('seaborn-deep')
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.figsize'] = (12, 8)

pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 400)
import warnings
warnings.filterwarnings('ignore')
import sklearn.base as skb
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import sklearn.utils as sku
import sklearn.linear_model as sklm
import sklearn.neighbors as skn
import sklearn.ensemble as ske
import catboost as cb
import scipy.stats as sstats
import random
seed = 12
np.random.seed(seed)

from datetime import date


# In[2]:


get_ipython().system('pip install pandas-profiling')
import pandas_profiling as pp


# In[3]:


# important funtions
def datasetShape(df):
    rows, cols = df.shape
    print("The dataframe has",rows,"rows and",cols,"columns.")
    
# select numerical and categorical features
def divideFeatures(df):
    numerical_features = df.select_dtypes(include=[np.number])
    categorical_features = df.select_dtypes(include=[np.object])
    return numerical_features, categorical_features


# In[4]:


df = pd.read_csv('train_Df64byy.csv')
df.head()


# In[5]:


df_test = pd.read_csv('test_YCcRUnU.csv')
df_test.head()


# In[6]:


# set target feature
targetFeature='Response'


# In[7]:


# check dataset shape
datasetShape(df)


# In[8]:


# remove ID from train data
df.drop(['ID'], inplace=True, axis=1)


# In[9]:


# check for duplicates
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)


# In[10]:


df.info()


# In[11]:


df_test.info()


# In[12]:


df.describe()


# # Step 2: EDA

# In[13]:


cont_features, cat_features = divideFeatures(df)
cat_features.head()


# ### Univariate Analysis

# In[14]:


# check target feature distribution
df[targetFeature].hist()
plt.show()


# In[15]:


# boxplots of numerical features for outlier detection

fig = plt.figure(figsize=(16,16))
for i in range(len(cont_features.columns)):
    fig.add_subplot(3, 3, i+1)
    sns.boxplot(y=cont_features.iloc[:,i])
plt.tight_layout()
plt.show()


# In[16]:


# distplots for categorical data

fig = plt.figure(figsize=(16,20))
for i in range(len(cat_features.columns)):
    fig.add_subplot(3, 3, i+1)
    cat_features.iloc[:,i].hist()
    plt.xlabel(cat_features.columns[i])
plt.tight_layout()
plt.show()


# In[17]:


# plot missing values

def calc_missing(df):
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing != 0]
    missing_perc = missing/df.shape[0]*100
    return missing, missing_perc

if df.isna().any().sum()>0:
    missing, missing_perc = calc_missing(df)
    missing.plot(kind='bar',figsize=(16,6))
    plt.title('Missing Values')
    plt.show()
else:
    print("No missing values")


# In[18]:


sns.pairplot(df)
plt.show()


# In[19]:


# correlation heatmap for all features
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, annot=True)
plt.show()


# ### Profiling for Whole Data

# In[20]:


profile = pp.ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_file("profile.html")


# In[21]:


profile.to_notebook_iframe()


# # Step 3: Data Preparation

# ### Skewness

# In[22]:


skewed_features = cont_features.apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_features


# ### Handle Missing

# In[23]:


# plot missing values

def calc_missing(df):
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing != 0]
    missing_perc = missing/df.shape[0]*100
    return missing, missing_perc

if df.isna().any().sum()>0:
    missing, missing_perc = calc_missing(df)
    missing.plot(kind='bar',figsize=(14,5))
    plt.title('Missing Values')
    plt.show()
else:
    print("No Missing Values")


# In[24]:


# remove all columns having no values
df.dropna(axis=1, how="all", inplace=True)
df.dropna(axis=0, how="all", inplace=True)
datasetShape(df)


# In[25]:


# def fillNan(df, col, value):
#     df[col].fillna(value, inplace=True)


# In[26]:


# # setting missing values to most occurring values
# fillNan(df, 'Health Indicator', df['Health Indicator'].mode()[0])
# fillNan(df_test, 'Health Indicator', df['Health Indicator'].mode()[0])
# df['Health Indicator'].isna().any()


# In[27]:


# # setting missing values to most occurring values
# # try changing with ML algo for missing
# fillNan(df, 'Holding_Policy_Duration', df['Holding_Policy_Duration'].mode()[0])
# fillNan(df_test, 'Holding_Policy_Duration', df['Holding_Policy_Duration'].mode()[0])
# df['Holding_Policy_Duration'].isna().any()


# In[28]:


# # setting missing values to most occurring values
# # try changing with ML algo for missing
# fillNan(df, 'Holding_Policy_Type', df['Holding_Policy_Type'].mode()[0])
# fillNan(df_test, 'Holding_Policy_Type', df['Holding_Policy_Type'].mode()[0])
# df['Holding_Policy_Type'].isna().any()


# ### Health Indicator Missing Prediction

# In[29]:


# # convert city code to int after removing C from it
# df['City_Code'] = pd.to_numeric(df['City_Code'].map(lambda x:x[1:]))
# df_test['City_Code'] = pd.to_numeric(df_test['City_Code'].map(lambda x:x[1:]))
# df['City_Code'].head()


# In[30]:


cont_features, cat_features = divideFeatures(df)
cont_features.columns.tolist()


# In[31]:


# get all not null records for imputing
X_impute = df[df['Health Indicator'].isna()==False]
y_impute = X_impute.pop('Health Indicator')

# remove categorical cols and targetFeature from X_impute
X_impute = X_impute[cont_features.columns.tolist()]
X_impute.drop(['Holding_Policy_Type', targetFeature], inplace=True, axis=1)

# impute with CatBoostClassifier
imputer_model = cb.CatBoostClassifier(random_state=seed, verbose=0)
imputer_model.fit(X_impute, y_impute)


# In[32]:


# predict values for train section
X_test_impute = df[df['Health Indicator'].isna()==True]
X_test_impute = X_test_impute[X_impute.columns.tolist()]
y_test_impute = imputer_model.predict(X_test_impute)

# setting value after prediction in df
for i,x in enumerate(X_test_impute.index):
    df.loc[x,'Health Indicator'] = y_test_impute[i]
    
# predict values for test section
X_test_impute = df_test[df_test['Health Indicator'].isna()==True]
X_test_impute = X_test_impute[X_impute.columns.tolist()]
y_test_impute = imputer_model.predict(X_test_impute)

# setting value after prediction in df
for i,x in enumerate(X_test_impute.index):
    df_test.loc[x,'Health Indicator'] = y_test_impute[i]


# ### Holding_Policy_Duration Missing Prediction

# In[33]:


# # convert Health Indicator to int after removing X from it
# df['Health Indicator'] = pd.to_numeric(df['Health Indicator'].map(lambda x:x[1:]))
# df_test['Health Indicator'] = pd.to_numeric(df_test['Health Indicator'].map(lambda x:x[1:]))
# df['Health Indicator'].head()


# In[34]:


cont_features, cat_features = divideFeatures(df)
cont_features.columns.tolist()


# In[35]:


# get all not null records for imputing
X_impute = df[df['Holding_Policy_Duration'].isna()==False]
y_impute = X_impute.pop('Holding_Policy_Duration')

# remove categorical cols and targetFeature from X_impute
X_impute = X_impute[cont_features.columns.tolist()]
X_impute.drop(['Holding_Policy_Type', targetFeature], inplace=True, axis=1)

# impute with RandomForestClassifier
imputer_model = cb.CatBoostClassifier(random_state=seed, verbose=0)
imputer_model.fit(X_impute, y_impute)


# In[36]:


# predict values for train section
X_test_impute = df[df['Holding_Policy_Duration'].isna()==True]
X_test_impute = X_test_impute[X_impute.columns.tolist()]
y_test_impute = imputer_model.predict(X_test_impute)

# setting value after prediction in df
for i,x in enumerate(X_test_impute.index):
    df.loc[x,'Holding_Policy_Duration'] = y_test_impute[i]
    
# predict values for test section
X_test_impute = df_test[df_test['Holding_Policy_Duration'].isna()==True]
X_test_impute = X_test_impute[X_impute.columns.tolist()]
y_test_impute = imputer_model.predict(X_test_impute)

# setting value after prediction in df
for i,x in enumerate(X_test_impute.index):
    df_test.loc[x,'Holding_Policy_Duration'] = y_test_impute[i]


# ### Holding_Policy_Type Missing Prediction

# In[37]:


# get all not null records for imputing
X_impute = df[df['Holding_Policy_Type'].isna()==False]
y_impute = X_impute.pop('Holding_Policy_Type')

# remove categorical cols and targetFeature from X_impute
cols_impute = cont_features.columns.tolist()
cols_impute.remove('Holding_Policy_Type')
X_impute = X_impute[cols_impute]
X_impute.drop([targetFeature], inplace=True, axis=1)

# impute with RandomForestClassifier
imputer_model = cb.CatBoostClassifier(random_state=seed, verbose=0)
imputer_model.fit(X_impute, y_impute)


# In[38]:


# predict values for train section
X_test_impute = df[df['Holding_Policy_Type'].isna()==True]
X_test_impute = X_test_impute[X_impute.columns.tolist()]
y_test_impute = imputer_model.predict(X_test_impute)

# setting value after prediction in df
for i,x in enumerate(X_test_impute.index):
    df.loc[x,'Holding_Policy_Type'] = y_test_impute[i]
    
# predict values for test section
X_test_impute = df_test[df_test['Holding_Policy_Type'].isna()==True]
X_test_impute = X_test_impute[X_impute.columns.tolist()]
y_test_impute = imputer_model.predict(X_test_impute)

# setting value after prediction in df
for i,x in enumerate(X_test_impute.index):
    df_test.loc[x,'Holding_Policy_Type'] = y_test_impute[i]


# In[39]:


print("Train Missing:",df.isna().any().sum())
print("Test Missing:",df_test.isna().any().sum())


# ## Derive Features

# In[40]:


# feature for age difference between Upper_Age and Lower_Age
df['age_diff'] = abs(df['Upper_Age'] - df['Lower_Age'])
df_test['age_diff'] = abs(df_test['Upper_Age'] - df_test['Lower_Age'])
df_test.head()


# In[41]:


# drop Lower_Age column as it is highly correlated with Upper_age and we also have its info in age_diff
df.drop('Lower_Age', axis=1, inplace=True)
df_test.drop('Lower_Age', axis=1, inplace=True)
df_test.head()


# ## Create Dummy Features

# In[42]:


df['Holding_Policy_Duration'] = pd.to_numeric(df['Holding_Policy_Duration'].map(lambda x:'15' if x == '14+' else x))
df_test['Holding_Policy_Duration'] = pd.to_numeric(df_test['Holding_Policy_Duration'].map(lambda x:'15' if x == '14+' else x))
df_test['Holding_Policy_Duration'].head()


# In[43]:


cont_features, cat_features = divideFeatures(df)
cat_features


# In[44]:


# label encoding on categorical features
def mapFeature(data, f, data_test=None):
    feat = data[f].unique()
    feat_idx = [x for x in range(len(feat))]

    data[f].replace(feat, feat_idx, inplace=True)
    if data_test is not None:
        data_test[f].replace(feat, feat_idx, inplace=True)


# In[45]:


for col in cat_features.columns:
    mapFeature(df, col, df_test)
df_test.head()


# ### One-Hot Encoding

# In[46]:


# extract numerical and categorical for dummy and scaling later
custom_feat = ['City_Code', 'Health Indicator']
# custom_feat = ['Health Indicator']
for feat in cat_features.columns:
    if len(df[feat].unique()) > 2 and feat in custom_feat:
        dummyVars = pd.get_dummies(df[feat], drop_first=True, prefix=feat+"_")
        df = pd.concat([df, dummyVars], axis=1)
        df.drop(feat, axis=1, inplace=True)
datasetShape(df)

df.head()


# In[47]:


# extract numerical and categorical for dummy and scaling later
custom_feat = ['City_Code', 'Health Indicator']
# custom_feat = ['Health Indicator']
for feat in cat_features.columns:
    if len(df_test[feat].unique()) > 2 and feat in custom_feat:
        dummyVars = pd.get_dummies(df_test[feat], drop_first=True, prefix=feat+"_")
        df_test = pd.concat([df_test, dummyVars], axis=1)
        df_test.drop(feat, axis=1, inplace=True)
datasetShape(df_test)

df_test.head()


# In[48]:


# # dropping holding policy features
# df.drop(['Holding_Policy_Duration', 'Holding_Policy_Type'], inplace=True, axis=1)
# df_test.drop(['Holding_Policy_Duration', 'Holding_Policy_Type'], inplace=True, axis=1)


# # Step 4: Data Modelling
# 
# ### Split Train-Test Data

# In[49]:


# helper functions

def log1p(vec):
    return np.log1p(abs(vec))

def expm1(x):
    return np.expm1(x)

def clipExp(vec):
    return np.clip(expm1(vec), 0, None)

def printScore(y_train, y_train_pred):
    print(skm.roc_auc_score(y_train, y_train_pred))


# In[50]:


# shuffle samples
df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

df_y = df_shuffle.pop(targetFeature)
df_X = df_shuffle

# split into train dev and test
X_train, X_test, y_train, y_test = skms.train_test_split(df_X, df_y, train_size=0.8, random_state=seed)
print(f"Train set has {X_train.shape[0]} records out of {len(df_shuffle)} which is {round(X_train.shape[0]/len(df_shuffle)*100)}%")
print(f"Test set has {X_test.shape[0]} records out of {len(df_shuffle)} which is {round(X_test.shape[0]/len(df_shuffle)*100)}%")


# ### Feature Scaling

# In[51]:


# scaler = skp.RobustScaler()
# scaler = skp.MinMaxScaler()
scaler = skp.StandardScaler()

# apply scaling to all numerical variables except dummy variables as they are already between 0 and 1
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# scale test data with transform()
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

# view sample data
X_train.describe()


# ## Model Building

# In[52]:


# X_train_small = X_train.sample(frac=0.3)
# y_train_small = y_train.iloc[X_train_small.index.tolist()]
# X_train_small.shape


# In[53]:


class_weights = sku.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights = dict(enumerate(class_weights))
class_weights


# In[54]:


sample_weights = sku.class_weight.compute_sample_weight('balanced', y_train)
sample_weights


# ### KNN

# In[55]:


knn = skn.KNeighborsClassifier(n_neighbors = 5, n_jobs=-1)
knn.fit(X_train, y_train)

# predict
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ### Logistic Regression

# In[56]:


log_model = sklm.LogisticRegression()
log_model.fit(X_train, y_train, sample_weight=sample_weights)

# predict
y_train_pred = log_model.predict(X_train)
y_test_pred = log_model.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# In[57]:


enet_model = sklm.ElasticNetCV(l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                    alphas = [1, 0.1, 0.01, 0.001, 0.0005], cv=10)
enet_model.fit(X_train, y_train)

# predict
y_train_pred = enet_model.predict(X_train)
y_test_pred = enet_model.predict(X_test)
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# In[58]:


ridge_model = sklm.RidgeCV(scoring = "neg_mean_squared_error", 
                    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0, 10], cv=5
                   )
ridge_model.fit(X_train, y_train)

# predict
y_train_pred = ridge_model.predict(X_train)
y_test_pred = ridge_model.predict(X_test)
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ### CatBoost

# In[59]:


import catboost as cb

cat_model = cb.CatBoostClassifier(loss_function='Logloss', verbose=0, eval_metric='AUC', class_weights=class_weights,
                           use_best_model=True, iterations=500)
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
print(cat_model.best_score_)

y_train_pred = cat_model.predict(X_train)
y_test_pred = cat_model.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ### Gradient Boosting

# In[60]:


# # Grid used
# param_test1 = {
#     'n_estimators': [10, 50, 100, 500],
#     'max_depth': np.arange(2, 12, 2)
# }
# gb_cv1 = skms.GridSearchCV(estimator = ske.GradientBoostingClassifier(loss='deviance', random_state=seed), 
#                              param_grid = param_test1, n_jobs=-1, 
#                              cv=5, verbose=1)
# # gb_cv1.fit(X_train_small, y_train_small)
# gb_cv1.fit(X_train, y_train, sample_weight=sample_weights)
# print(gb_cv1.best_params_, gb_cv1.best_score_)
# # n_estimators = 1000
# # max_depth = 10


# In[61]:


# # Grid used
# param_test2 = {
#     'min_samples_split': np.arange(2, 12, 3),
#     'min_samples_leaf': np.arange(1, 10, 3)
# }
# gb_cv2 = skms.GridSearchCV(estimator = ske.GradientBoostingClassifier(loss='deviance', random_state=seed,
#                                                                  n_estimators=50,
#                                                                  max_depth=7), 
#                              param_grid = param_test2, n_jobs=-1, 
#                              cv=5, verbose=1)
# gb_cv2.fit(X_train, y_train)
# print(gb_cv2.best_params_, gb_cv2.best_score_)
# print(gb_cv2.best_estimator_)
# # min_samples_split = 8
# # min_samples_leaf = 1


# In[62]:


gb_model = ske.GradientBoostingClassifier(loss='deviance', random_state=seed, verbose=0,
                                    n_estimators=50, max_depth=7,
                                    min_samples_leaf=1, min_samples_split=8)
gb_model.fit(X_train, y_train, sample_weight=sample_weights)

# predict
y_train_pred = gb_model.predict(X_train)
y_test_pred = gb_model.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ### Extra Trees

# In[63]:


# # Grid used
# param_test1 = {
#     'n_estimators': [10, 50, 100, 500, 1000],
#     'max_depth': np.arange(2, 12, 2)
# }
# extra_cv1 = skms.GridSearchCV(estimator = ske.ExtraTreesClassifier(criterion='gini', random_state=seed), 
#                              param_grid = param_test1, scoring='neg_mean_squared_error', n_jobs=-1, 
#                              cv=5, verbose=1)
# # extra_cv1.fit(X_train_small, y_train_small)
# extra_cv1.fit(X_train, y_train)
# print(extra_cv1.best_params_, extra_cv1.best_score_)
# # n_estimators = 200
# # max_depth = 10


# In[64]:


# # Grid used
# param_test2 = {
#     'min_samples_split': np.arange(5, 18, 3),
#     'min_samples_leaf': np.arange(1, 10, 2)
# }
# extra_cv2 = skms.GridSearchCV(estimator = ske.ExtraTreesClassifier(criterion='gini', random_state=seed,
#                                                                  n_estimators=200,
#                                                                  max_depth=10), 
#                               param_grid = param_test2, scoring='neg_mean_squared_error', n_jobs=-1, 
#                               cv=5, verbose=1)
# extra_cv2.fit(X_train, y_train)
# print(extra_cv2.best_params_, extra_cv2.best_score_)
# print(extra_cv2.best_estimator_)
# # min_samples_split = 5
# # min_samples_leaf = 1


# In[65]:


extra_model = ske.ExtraTreesClassifier(criterion='gini', random_state=1, verbose=0, n_jobs=-1,
                              n_estimators=200,max_depth=10,
                              min_samples_split = 5, min_samples_leaf = 1)
extra_model.fit(X_train, y_train, sample_weight=sample_weights)

# predict
y_train_pred = extra_model.predict(X_train)
y_test_pred = extra_model.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ### AdaBoost

# In[66]:


ada_model = ske.AdaBoostClassifier(random_state=1)
ada_model.fit(X_train, y_train, sample_weight=sample_weights)

# predict
y_train_pred = ada_model.predict(X_train)
y_test_pred = ada_model.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ### RandomForest

# In[67]:


rf_model = ske.RandomForestClassifier(verbose=0, random_state=1, n_jobs=-1, class_weight='balanced_subsample',
                                 n_estimators=200,max_depth=10, 
                                 min_samples_split = 7, min_samples_leaf = 1
                                )
rf_model.fit(X_train, y_train)

# predict
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ### XGBoost

# In[68]:


import xgboost as xg


# In[69]:


# # Grid used
# param_test1 = {
#     'max_depth': np.arange(5, 12, 2),
#     'learning_rate': np.arange(0.04, 0.07, 0.01)
# }
# xgb_cv1 = skms.GridSearchCV(estimator = xg.XGBClassifier(n_estimators=100, objective='reg:squarederror', nthread=4, seed=seed), 
#                              param_grid = param_test1, scoring='neg_mean_squared_error', n_jobs=4, 
#                              iid=False, cv=5, verbose=1)
# xgb_cv1.fit(X_train_small, y_train_small)
# print(xgb_cv1.best_params_, xgb_cv1.best_score_)
# # max_depth = 10
# # learning_rate = 0.04


# In[70]:


# param_test2 = {
#  'subsample': np.arange(0.5, 1, 0.1),
#  'min_child_weight': range(1, 6, 1)
# }
# xgb_cv2 = skms.GridSearchCV(estimator = xg.XGBClassifier(n_estimators=500, max_depth = 10, 
#                                                      objective= 'reg:squarederror', nthread=4, seed=seed), 
#                             param_grid = param_test2, scoring='neg_mean_squared_error', n_jobs=4,
#                             cv=5, verbose=1)
# xgb_cv2.fit(X_train_small, y_train_small)
# print(xgb_cv2.best_params_, xgb_cv2.best_score_)
# print(xgb_cv2.best_estimator_)
# # subsample = 0.5
# # min_child_weight = 2


# In[71]:


# working without scaling
xgb_model = xg.XGBClassifier(objective ='binary:logistic', random_state=seed, verbose=0,
                      n_estimators=500, max_depth = 10)
xgb_model.fit(X_train, y_train)

# predict
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ### LightGBM

# In[72]:


import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(objective='binary', class_weight=class_weights, random_state=1, n_jobs=-1,
                         n_estimators=50)
lgb_model.fit(X_train, y_train)

# predict
y_train_pred = lgb_model.predict(X_train)
y_test_pred = lgb_model.predict(X_test)
print(skm.accuracy_score(y_train, y_train_pred))
print(skm.accuracy_score(y_test, y_test_pred))
printScore(y_train, y_train_pred)
printScore(y_test, y_test_pred)


# ## Deep Learning Model

# In[ ]:





# In[74]:


import tensorflow as tf
import tensorflow_addons as tfa
print("TF version:-", tf.__version__)
import keras as k
tf.random.set_seed(seed)


# In[ ]:


THRESHOLD = .999
bestModelPath = './best_model.hdf5'

class myCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > THRESHOLD):
            print("\n\nStopping training as we have reached our goal.")   
            self.model.stop_training = True

mycb = myCallback()
checkpoint = k.callbacks.ModelCheckpoint(filepath=bestModelPath, monitor='val_loss', verbose=1, save_best_only=True)

callbacks_list = [mycb,
                  checkpoint
                 ]
            
def plotHistory(history):
    print("Min. Validation ACC Score",min(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()


# In[ ]:


epochs = 40

model_1 = k.models.Sequential([
    k.layers.Dense(2048, activation='relu', input_shape=(X_train.shape[1],)),
#     k.layers.Dropout(0.3),
    
    k.layers.Dense(1024, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(512, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(1, activation='sigmoid'),
])
print(model_1.summary())

model_1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[
#                   tfa.metrics.F1Score(num_classes=1),
                  'accuracy'
              ]
)
history = model_1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, 
                      batch_size=2048, 
#                       class_weight=class_weights,
                      callbacks=[callbacks_list]
                     )


# In[ ]:


plotHistory(history)


# In[ ]:


# y_train_pred = model_1.predict(X_train)
# y_test_pred = model_1.predict(X_test)
# print(skm.accuracy_score(y_train, y_train_pred))
# print(skm.accuracy_score(y_test, y_test_pred))
# printScore(y_train, y_train_pred)
# printScore(y_test, y_test_pred)


# # Test Evaluation & Submission

# In[ ]:


# Generate Ensembles

# def rmse_cv(model):
#     '''
#     Use this function to get quickly the rmse score over a cv
#     '''
#     rmse = np.sqrt(-skms.cross_val_score(model, X_train, y_train, 
#                                          scoring="neg_mean_squared_error", cv = 5, n_jobs=-1))
#     return rmse

# class MixModel(skb.BaseEstimator, skb.RegressorMixin, skb.TransformerMixin):
#     '''
#     Here we will get a set of models as parameter already trained and 
#     will calculate the mean of the predictions for using each model predictions
#     '''
#     def __init__(self, algs):
#         self.algs = algs

#     # Define clones of parameters models
#     def fit(self, X, y):
#         self.algs_ = [skb.clone(x) for x in self.algs]
        
#         # Train cloned base models
#         for alg in self.algs_:
#             alg.fit(X, y)

#         return self
    
#     # Average predictions of all cloned models
#     def predict(self, X):
#         predictions = np.column_stack([
#             stacked_model.predict(X) for stacked_model in self.algs_
#         ])
#         return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)


# In[ ]:


# mixed_model = MixModel(algs = [
# #     ridge_model, 
# #     enet_model, 
# #     extra_model, 
# #     cat_model,
# #     rf_model,
# #     xgb_model,
# #     gb_model,
# #     lgb_model,
#     ada_model
# ])
# # score = rmse_cv(mixed_model)
# # print("\nAveraged base algs score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

# mixed_model.fit(X_train, y_train)

# # predict
# y_train_pred = mixed_model.predict(X_train)
# y_test_pred = mixed_model.predict(X_test)
# printScore(y_train, y_train_pred)
# printScore(y_test, y_test_pred)


# In[ ]:


def getTestResults(m=None):
    df_final = df.sample(frac=1, random_state=1).reset_index(drop=True)
    test_cols = [x for x in df_final.columns if targetFeature not in x]
    df_final_test = df_test[test_cols]
    df_y = df_final.pop(targetFeature)
    df_X = df_final

#     scaler = skp.RobustScaler()
#     scaler = skp.MinMaxScaler()
    scaler = skp.StandardScaler()

    df_X = pd.DataFrame(scaler.fit_transform(df_X), columns=df_X.columns)
    df_final_test = pd.DataFrame(scaler.transform(df_final_test), columns=df_X.columns)

    sample_weights = sku.class_weight.compute_sample_weight('balanced', df_y)
    
    if m is None:

#         lmr = sklm.LogisticRegression()
#         lmr.fit(df_X, df_y)

        lmr = cb.CatBoostClassifier(loss_function='Logloss', verbose=0, eval_metric='AUC', class_weights=class_weights)
        lmr.fit(df_X, df_y)

#         lmr = ske.ExtraTreesClassifier(criterion='gini', random_state=1, verbose=0, n_jobs=-1,
#                               n_estimators=200,max_depth=10, min_samples_split = 5, min_samples_leaf = 1)
#         lmr.fit(df_X, df_y, sample_weight=sample_weights)

#         lmr = ske.AdaBoostClassifier(random_state=seed)
#         lmr.fit(df_X, df_y, sample_weight=sample_weights)

#         lmr = ske.GradientBoostingClassifier(loss='deviance', random_state=seed, verbose=0,
#                                     n_estimators=50, max_depth=7,min_samples_leaf=1, min_samples_split=8)
#         lmr.fit(df_X, df_y, sample_weight=sample_weights)

#         lmr = ske.RandomForestClassifier(verbose=0, random_state=1, n_jobs=-1, class_weight='balanced_subsample',
#                                  n_estimators=200,max_depth=10, min_samples_split = 7, min_samples_leaf = 1)
#         lmr.fit(df_X, df_y)

#         lmr = xg.XGBClassifier(objective ='binary:logistic', random_state=seed, verbose=0,
#                       n_estimators=500, max_depth = 10)
#         lmr.fit(df_X, df_y)

#         lmr = lgb.LGBMClassifier(objective='binary', class_weight=class_weights, random_state=1, n_jobs=-1, n_estimators=50)
#         lmr.fit(df_X, df_y)

    else:
        lmr = m

    # predict
    y_train_pred = lmr.predict(df_X)
    y_test_pred = lmr.predict(df_final_test)
    if m is not None:
        y_train_pred = [round(y[0]) for y in y_train_pred]
        y_test_pred = [round(y[0]) for y in y_test_pred]
    print(skm.accuracy_score(df_y, y_train_pred))
    printScore(df_y, y_train_pred)
    return y_test_pred

# ML models
results = getTestResults()

# Neural Network model
# results = getTestResults(k.models.load_model(bestModelPath))


# In[ ]:


submission = pd.DataFrame({
    'ID': df_test['ID'],
    targetFeature: results,
})
print(submission.Response.value_counts())
submission.head()


# In[ ]:


submission.to_csv('./submission_Cat-robust.csv', index=False)

