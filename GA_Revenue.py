import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import collections
from sklearn.preprocessing import LabelEncoder
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
import lightgbm as lgbm
#import dask.dataframe as dd
import gc
gc.enable()

train = pd.read_csv('E:/College/Analytics/Python/GA-Revenue/train_v2.csv')
test = pd.read_csv('E:/College/Analytics/Python/GA-Revenue/test_v2.csv')
#test = dd.read_csv('E:/College/Analytics/Python/GA-Revenue/test_v2.csv')
#train = dd.read_csv('E:/College/Analytics/Python/GA-Revenue/train_v2.csv')

train.dtypes
train.head(5)
train.shape
train.columns.values

test.dtypes
test.head(5)
test.shape
test.columns.values

train.isnull().sum().sort_values(ascending=False)

train['channelGrouping'].value_counts().plot(kind="bar",title="Channel Grouping Distrubution",figsize=(8,8),rot=25)
sns.countplot(train['channelGrouping'])

train['socialEngagementType'].value_counts()
train['socialEngagementType'].describe()

train.head(1)[["date","visitStartTime"]]
train["date"]=pd.to_datetime(train["date"],format="%Y%m%d")
train["visitStartTime"]=pd.to_datetime(train["visitStartTime"],unit='s')

train.head(1)[["date","visitStartTime"]]

list_of_devices = train['geoNetwork'].apply(json.loads).tolist()
keys = []
for devices_iter in list_of_devices:
    for list_element in list(devices_iter.keys()):
        if list_element not in keys:
            keys.append(list_element)
            
"keys existed in device attribute are:{}".format(keys)

device_df = pd.DataFrame(train.device.apply(json.loads).tolist())[["browser","operatingSystem","deviceCategory","isMobile"]]
device_df.head(5)
traffic_source_df = pd.DataFrame(train.trafficSource.apply(json.loads).tolist())[["keyword","medium" , "source"]]
traffic_source_df.head(5)
geo_df = pd.DataFrame(train.geoNetwork.apply(json.loads).tolist())[["continent","subContinent","country","city"]]
geo_df.head(5)
totals_df = pd.DataFrame(train.totals.apply(json.loads).tolist())[["transactionRevenue", "newVisits", "bounces", "pageviews", "hits"]]
totals_df.head(5)

#Data Manipulation/Vizualization
sns.countplot(device_df['isMobile'])
sns.countplot(device_df['deviceCategory'])
device_df['browser'].value_counts().head(10).plot(kind="bar",title="Browser Distrubution",figsize=(8,8),rot=25)
device_df['operatingSystem'].value_counts().head(10).plot(kind="bar",title="OS Distrubution",figsize=(8,8),rot=25,color='teal')

plt.subplots(figsize=(7, 6))
sns.countplot(geo_df[geo_df['continent']== "Asia"]['subContinent'])

geo_df['continent'].value_counts().plot(kind="bar",title="Continent Distrubution",figsize=(8,8),rot=0)
geo_df[geo_df['continent']== "Asia"]['subContinent'].value_counts().plot(kind="bar",title="Asia Distrubution",figsize=(8,8),rot=0)
geo_df[geo_df['continent']== "Europe"]['subContinent'].value_counts().plot(kind="bar",title="Europe Distrubution",figsize=(8,8),rot=0)

traffic_source_df["medium"].value_counts().plot(kind="bar",title="Medium",rot=0)
traffic_source_df["source"].value_counts().head(10).plot(kind="bar",title="source",rot=75,color="teal")

fig,axes = plt.subplots(1,2,figsize=(15,10))
traffic_source_df["keyword"].value_counts().head(10).plot(kind="bar",ax=axes[0], title="keywords (total)",color="yellow")
traffic_source_df[traffic_source_df["keyword"] != "(not provided)"]["keyword"].value_counts().head(15).plot(kind="bar",ax=axes[1],title="keywords (dropping NA)",color="c")

train["revenue"] = pd.DataFrame(train.totals.apply(json.loads).tolist())[["transactionRevenue"]]
train["revenue"].value_counts().sort_values(ascending=False)
#data["revenue"]=data["revenue"].astype(np.int64)

revdat_df = train[["revenue", "date","visitNumber"]].dropna()
revdat_df["revenue"] = revdat_df.revenue.astype(np.int64)
revdat_df.head()
plt.subplots(figsize=(20, 10))
plt.plot(revdat_df.groupby("date")["revenue"].sum())

ab=revdat_df.groupby("date").sum()

visitdate_df = train[["date","visitNumber"]]
visitdate_df["visitNumber"] = visitdate_df.visitNumber.astype(np.int64)
visitdate_df.groupby("date").sum()

fig, ax1 = plt.subplots(figsize=(20,10))
t = ab.index
s1 = ab["visitNumber"]
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('day')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('visitNumber', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
s2 = ab["revenue"]
ax2.plot(t, s2, 'r--')
ax2.set_ylabel('revenue', color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()

fig,ax = plt.subplots(figsize=(9,5))
ax.set_title("Histogram of log(visitNumbers) \n per session")
ax.set_ylabel("Repetition")
ax.set_xlabel("Log(visitNumber)")
ax.grid(color='b', linestyle='-', linewidth=0.1)
ax.hist(np.log(train['visitNumber']))

tmp_least_visitNumbers_list = collections.Counter(list(train.visitNumber)).most_common()[:-10-1:-1]
tmp_most_visitNumbers_list = collections.Counter(list(train.visitNumber)).most_common(10)
least_visitNumbers = []
most_visitNumbers = []
for i in tmp_least_visitNumbers_list:
    least_visitNumbers.append(i[0])
for i in tmp_most_visitNumbers_list:
    most_visitNumbers.append(i[0])
"10 most_common visitNumbers are {} times and 10 least_common visitNumbers are {} times".format(most_visitNumbers,least_visitNumbers)

train_all=pd.concat([train.drop(["hits"],axis=1),device_df,geo_df,traffic_source_df,totals_df],axis=1)
train_all.dtypes
from datetime import datetime
train_all["month"] = train_all['date'].dt.month
train_all['visitHour'] = (train_all['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
plt.figure(figsize=(10,5))
sns.barplot(x='month', y=train_all['transactionRevenue'].astype(np.float), data=train_all)
plt.figure(figsize=(10,5))
sns.barplot(x='visitHour', y=train_all['transactionRevenue'].astype(np.float), data=train_all)
train_all.visitNumber.value_counts().head(10).plot(kind="bar",title="Vistor Numbers Distrubution",figsize=(8,8),rot=25,color='teal')

df_train = train_all.drop(['date','month','device','geoNetwork','trafficSource','totals','customDimensions', 'socialEngagementType', 'visitStartTime', 'visitId', 'fullVisitorId' , 'revenue'], axis=1)
df_train.dtypes
df_train.shape
df_train.isnull().sum().sort_values(ascending=False)
df_train=df_train.fillna(0)

numerical_features = ['transactionRevenue','visitNumber', 'newVisits', 'bounces', 'pageviews', 'hits']

for col in numerical_features:
    df_train[col] = df_train[col].astype(np.float)

vst_rev=df_train.groupby('visitNumber')['transactionRevenue'].agg(['count','mean','sum'])
vst_rev.columns = ["count", "mean transaction","total revenue"]
vst_rev = vst_rev.sort_values(by="count", ascending=False)
sns.barplot(y=vst_rev['total revenue'].head(10),x=vst_rev.index[:10])
sns.barplot(y=vst_rev['mean transaction'].head(10),x=vst_rev.index[:10])


def feat_plot(col):
    pt = df_train.loc[:,[col, 'transactionRevenue']]
    feat_vis=pt.groupby(col)['transactionRevenue'].agg(['count','mean'])
    feat_vis.columns = ["count", "mean transaction value"]
    feat_vis['total_revenue'] = feat_vis['count']*feat_vis['mean transaction value']
    feat_vis = feat_vis.sort_values(by="count", ascending=False)
    plt.figure(figsize=(8, 16)) 
    plt.subplot(2,1,1)
    sns.barplot(x=feat_vis['count'].head(10), y=feat_vis.index[:10])
    plt.subplot(2,1,2)
    sns.barplot(x=feat_vis['mean transaction value'].head(10), y=feat_vis.index[:10])

feat_plot('browser')
feat_plot('continent')
feat_plot('country')
feat_plot('operatingSystem')
feat_plot('source')

from wordcloud import WordCloud
source = df_train['source']
wordcloud2 = WordCloud(width=800, height=400).generate(' '.join(source))
plt.figure( figsize=(12,10) )
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

# Model

categorical_features = ['channelGrouping', 'browser', 'operatingSystem', 'deviceCategory', 'isMobile',
                        'continent', 'subContinent', 'country', 'city', 'keyword', 'medium', 'source']    

for col in categorical_features:
    lbl = LabelEncoder()
    lbl.fit(list(df_train[col].values.astype('str')))
    df_train[col] = lbl.transform(list(df_train[col].values.astype('str')))


cat_feature=(df_train.dtypes == object) | (df_train.dtypes == bool)
cat_cols = df_train.columns[cat_feature].tolist()
le=LabelEncoder()
le.fit(list(df_train[cat_cols].values.astype('str')))
df_train[cat_cols] = df_train[cat_cols].apply(lambda col:le.fit_transform(col).values.astype('str'))

X=df_train.drop(['transactionRevenue'], axis=1)
y=np.log1p(df_train['transactionRevenue'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7, 
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
    }


lgtrain = lgbm.Dataset(X_train, label=y_train)
lgval = lgbm.Dataset(X_test, label=y_test)
lgb_model = lgbm.train(params, lgtrain, valid_sets=[lgval], num_boost_round=2000, early_stopping_rounds=100, verbose_eval=100)

pred_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

fig, ax = plt.subplots(figsize=(8,12))
lgbm.plot_importance(lgb_model, max_num_features=30, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
from sklearn.model_selection import GridSearchCV

grid= {"min_child_weight":[4,5,6],"max_depth":[-1,1,3,5], "learning_rate":[0.1,0.01,0.2]}
lgb=LGBMRegressor(random_state=96,objective='regression',metric='rmse')
gridsearch= GridSearchCV(lgb,param_grid=grid,cv=5)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_score_)
print(gridsearch.best_params_)

lgb= LGBMRegressor(objective='regression',metric='rmse',learning_rate=0.1,min_child_weight=4)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)
accuracy = lgb.score(X_test,y_test)
'Accuracy: ' + str(np.round(accuracy*100, 2)) + '%'
mean_absolute_error(y_test, lgb_pred)
mean_squared_error(y_test, lgb_pred)
np.sqrt(mean_squared_error(y_test, lgb_pred))

coefs = pd.Series(lgb.feature_importances_, index = X_train.columns)
imp_coefs = pd.concat([coefs.sort_values().head(10),coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.xlabel("LGB coefficient", weight='bold')
plt.title("Feature importance in the LightGB Model", weight='bold')
plt.show()

###################################################

#pip install bayesian-optimization

from bayes_opt import BayesianOptimization

def lgb_eval(num_leaves, num_iterations, feature_fraction,learning_rate,  bagging_fraction,bagging_frequency, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):        
        params = {'application':'regression_l2','metric':'rmse', 'early_stopping_round':100}
        
        params["num_leaves"] = int(round(num_leaves))
        params["num_iterations"] = int(num_iterations)
        params["learning_rate"] = learning_rate
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['bagging_frequency'] = bagging_frequency 
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        
        dtrain = lgbm.Dataset(data=X_train, label=y_train, categorical_feature = categorical_features, free_raw_data=False)
        cv_result = lgbm.cv(params, dtrain,nfold=5, verbose_eval=200,stratified=False)    
        
        #print(cv_result)
        # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
        return -1.0 * cv_result['rmse-mean'][-1]


lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (30, 200),
                                        'feature_fraction': (0.1, 0.9),
                                        'learning_rate' : (0.0001,0.01),
                                        'bagging_fraction': (0.8, 1),
                                        'bagging_frequency':(5,10),
                                        'num_iterations':(1000,5000),
                                        'max_depth': (5, 10),
                                        'lambda_l1': (0, 5),
                                        'lambda_l2': (0, 3),
                                        'min_split_gain': (0.001, 0.1),
                                        'min_child_weight': (5, 50)}, random_state=0)

# lgbBO.maximize(init_points=3, n_iter=5, acq='ei')

