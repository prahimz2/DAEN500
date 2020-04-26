#!/usr/bin/env python
# coding: utf-8

# In[6]:


# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:20:05 2020

@author: jakes
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn import preprocessing
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.feature_selection import RFECV
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score
import statsmodels.api as sm



stats = pd.read_csv('Season_Stats.csv')

# only grabbing years 2007-2018 (train on 2007-2017, test on 2018)
stats = stats[stats['Year'] > 2005].reset_index(drop=True) 

# dropping these two columns which are both entirely blank
stats = stats.drop(columns = ['blanl','blank2','Unnamed: 0'])


# In[7]:


stats.head()


# In[12]:


# group by year first, then player, then team
# this will let us see which players had multiple teams in a single year
# TOT value for teams is an aggregation of all the team data for the player
teams = stats.groupby(['Year','Player','Tm','ALL_STAR']).sum()
teams.head()


# In[13]:


# resetting the index will preserve the return order of the rows
# but the columns are treated a
teams = teams.reset_index()
teams.head(10)


# In[19]:


cols = teams.columns.values.tolist()


# In[15]:


# create a blank dataframe using the column values of Teams dataframe
# we are doing this so we can append values to it
TOT = pd.DataFrame(columns=cols)


# In[16]:


# this will give us all the rows where team == TOT
for i in range(len(teams)):
    if(teams['Tm'][i]=='TOT'):
        TOT.loc[i] = teams.loc[i]


# In[18]:


TOT.head()


# In[20]:


teams = teams.drop_duplicates(subset=['Player','Year'],keep=False)


# In[21]:


finalTeams = pd.concat([teams,TOT])
finalTeams = finalTeams.sort_values(by=['Year','Player'])
finalGrouped = finalTeams.groupby(['Year','Player','Tm','ALL_STAR']).sum()


# In[35]:


finalTeams = finalTeams.astype({'Year': 'int64', 'ALL_STAR':'bool','Age': 'int64','G':'int64','GS':'int64',
                                'MP':'int64','FG':'int64','FGA':'int64','3P':'int64','3PA':'int64',
                                '2P':'int64','2PA':'int64','FT':'int64','FTA':'int64','ORB':'int64',
                                'DRB':'int64','TRB':'int64','AST':'int64','STL':'int64','BLK':'int64',
                                'TOV':'int64','PF':'int64','PTS':'int64'})
finalTeams.dtypes


# In[22]:


finalTeams[finalTeams['Player']=='Isaiah Thomas']


# In[23]:


finalTeams.to_csv('Final_Season_Stats.csv')


# # EDA/ QA
#code for indexing using rows: df.loc[]

#Define function to accept a df and value as arguement and return a list of index positions of all occurences
def getIndexes(dfObj, value):

    listOfPos = list()

    result = dfObj.isin([value])

    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)

    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))

    return listOfPos

topOBPM = getIndexes(finalTeams, 47.8)
topWS = getIndexes(finalTeams, 20.3)
topeFG = getIndexes(finalTeams, 1.50000)
for i in range(len(topeFG)):
    print(i, topeFG[i])
    
# remove players where games < 10

teamsFiltered = finalTeams[finalTeams['G'] >= 10]

#remove players where minutes played <= 50

teamsFiltered = teamsFiltered[teamsFiltered['MP'] > 50]

teamsFiltered.describe() 

#create scatterplot for points scored and field goal %
plt.scatter(teamsFiltered['PTS'], teamsFiltered['FG%'])
plt.axhline(y=0.5, color='black')
plt.title('Field Goal Percentage vs Points Scored')

#create plot of All-star status and points scored
plt.scatter(teamsFiltered['PTS'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Points Scored')

#all-star vs age
plt.scatter(teamsFiltered['Age'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Age')

#all-star vs obpm
plt.scatter(teamsFiltered['OBPM'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs OBPM')

#all-star vs games
plt.scatter(teamsFiltered['G'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Games Played')

#all-star vs TOV
plt.scatter(teamsFiltered['TOV'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Turnovers')

#all-star vs WS
plt.scatter(teamsFiltered['WS/48'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Win Share')

#all-star vs BPM
plt.scatter(teamsFiltered['BPM'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs BPM')

#all-star vs PER
plt.scatter(teamsFiltered['PER'], teamsFiltered['ALL_STAR'])
plt.title('All-Star Status vs Player Efficiency Rating')

#remove players where minutes played < 200
teamsFiltered2 = teamsFiltered[teamsFiltered['MP'] >= 200]

#running previous charts on MP >= 200 df  (only %based charts effected)
#all-star vs obpm
plt.scatter(teamsFiltered2['OBPM'], teamsFiltered2['ALL_STAR'])
plt.title('All-Star Status vs OBPM')
plt.text(-5, 0.5, 'Minimum 200 minutes played')


#all-star vs BPM
plt.scatter(teamsFiltered2['BPM'], teamsFiltered2['ALL_STAR'])
plt.title('All-Star Status vs BPM')
plt.text(-5, 0.5, 'Minimum 200 minutes played')

#all-star vs PER
plt.scatter(teamsFiltered2['PTS'], teamsFiltered2['ALL_STAR'])
plt.title('All-Star Status vs Player Efficiency Rating')
plt.text(10, 0.5, 'Minimum 200 minutes played')

#dimension reduction using Recursive Feature Elimination
all_vars = ['Age', 'G', 'MP', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%',
            'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%',
            'OWS', 'DWS', 'WS', 'BPM', 'VORP', 'FG', 'FG%', '3P', '3P%',
            '2P', '2P%', 'eFG%', 'FT', 'FT%', 'TOV', 'PF', 'PTS']
all_vars_data = teamsFiltered2[all_vars]
all_vars_scaled = preprocessing.scale(all_vars_data)
y = teamsFiltered2.ALL_STAR
x_train, x_test, y_train, y_test = train_test_split(all_vars_scaled, y, test_size=0.3)
logmodel = LogisticRegression(max_iter=2000)
rfecv = RFECV(estimator=logmodel, step=1, scoring='accuracy')
rfecv.fit(x_train, y_train)
rfecv.support_
selected_vars = ['G', 'MP', '3PAr', 'TRB%', 'AST%', 'USG%', 'OWS', 'DWS', 'WS', 'BPM', 'VORP','3P']

print("Optimal number of features: %d" % rfecv.n_features_)
rfecv.support_

#perform logistic analysis using recursively selected vars
selected_vars_data = teamsFiltered2[selected_vars]
x_train2, x_test2, y_train2, y_test2 = train_test_split(selected_vars_data, y, test_size=0.3)
logmodel2 = LogisticRegression(max_iter=2000)
logmodel2.fit(x_train2, y_train2) 
predictions_2 = logmodel2.predict(x_test2)
print(classification_report(y_test2, predictions_2))
print(confusion_matrix(y_test2, predictions_2))
y_pred_proba = logmodel2.predict_proba(x_test2)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test2, y_pred_proba)
print('Train/Test split results:')
print(logmodel2.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test2, predictions_2))

print(logmodel2.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))


##jake's code
df = teamsFiltered2[teamsFiltered2['Year'] != 2017]
test_df = teamsFiltered2[teamsFiltered2['Year'] == 2017]
test_df.drop(test_df['ALL_STAR'])

df['player_index'] = df['Player'] + ': ' + df['Year'].astype(str)
df.set_index(df['player_index'], inplace = True)
df.drop(columns = ['player_index'], inplace = True)

drop = ['Player','Year','Tm','ALL_STAR','G','MP','GS','3PAr','FTr','WS/48','FG','FGA','3P','3PA','2P','2PA','FT','FTA']
features = df.drop(columns = drop).columns
x = df[features]
y = df['ALL_STAR']

x_scaled = preprocessing.scale(x)
print(x_scaled.shape)
y.shape
logreg=LogisticRegression(max_iter=200)

# Create the RFE object in order to determine the variables to keep
rfecv2 = RFECV(estimator=logreg, step=1, scoring='accuracy')
rfecv2.fit(x_scaled, y)
print("Optimal number of features: %d" % rfecv2.n_features_)
print('Selected features: %s' % list(x.columns[rfecv2.support_]))

plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv2.grid_scores_) + 1), rfecv2.grid_scores_)
plt.show()

selected_features = list(x.columns[rfecv2.support_])
x_filtered = x[selected_features]
x_filtered_scaled = preprocessing.scale(x_filtered)
y_final = df['ALL_STAR']

#descriptive statistics of scaled logistic Model
logit_model=sm.Logit(y_final,x_filtered_scaled)
result=logit_model.fit()
print(result.summary2())

#create training and test set with scaled data, train model, and test it
x_train3, x_test3, y_train3, y_test3 = train_test_split(x_filtered_scaled,y_final,test_size=0.3, random_state=0)
logreg_scaled = LogisticRegression()
model_res = logreg_scaled.fit(x_train3,y_train3)
logreg_scaled.fit(x_train3,y_train3)

y_pred = logreg.predict(x_test3)
print(classification_report(y_test3, y_pred))
print(confusion_matrix(y_test3, y_pred))

#Testing Logistic Regression
model_res.predict(x_train3)
x_train3['ALL_STAR_PROB'] = model_res.predict_proba(x_train3)[:,1]
x_train33_with_prob['ALL_STAR'] = y_train

#removing values with high p-values 
p_values = ['USG%', 'OWS', 'DBPM', 'BPM', 'VORP', 'AST', 'PF']
x_filtered_p = x_filtered[p_values]
x_filtered_p_scaled = preprocessing.scale(x_filtered_p)
x_train4, x_test4, y_train4, y_test4 = train_test_split(x_filtered_p_scaled,y_final,test_size=0.3, random_state=0)
logreg_filtered = LogisticRegression()
model_res_p = logreg_filtered.fit(x_train4,y_train4)
logreg_filtered.fit(x_train4,y_train4)

y_pred_p = logreg_filtered.predict(x_test4)
print(classification_report(y_test4, y_pred_p))
print(confusion_matrix(y_test4, y_pred_p))




#confusion matrix algorithm
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
cnf_matrix =    confusion_matrix(y_test2, predictions_2)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True, classes=['All-Star','Not-Star'],
                      title='Confusion matrix, with normalization')

tn, fp, fn, tp = confusion_matrix(y_test2, predictions_2).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
Accuracy = (tn+tp)*100/(tp+tn+fp+fn) 
print("Accuracy {:0.2f}%:".format(Accuracy))
#Precision 
Precision = tp/(tp+fp) 
print("Precision {:0.2f}".format(Precision))