#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df= pd.read_csv('Traffic_camera_offences_and_fines.csv')
df.head(5)


# In[2]:


df.tail(5)


# In[3]:


df.info(5)


# In[4]:


df.describe()


# In[11]:


df2 = df
for x in df['Offence_Month'].unique().tolist():
    y = x.split()
    if y[0] == "Jan":
        y[0] = "1"
    if y[0] == "Feb":
        y[0] = "2"
    if y[0] == "Mar":
        y[0] = "3"
    if y[0] == "Apr":
        y[0] = "4"
    if y[0] == "May":
        y[0] = "5"
    if y[0] == "Jun":
        y[0] = "6"
    if y[0] == "Jul":
        y[0] = '7'
    if y[0] == "Aug":
        y[0] = "8"
    if y[0] == "Sep":
        y[0] = "9"
    if y[0] == "Oct":
        y[0] = "10"
    if y[0] == "Nov":
        y[0] = "11"
    if y[0] == "Dec":
        y[0] = "12"
    df2 = df2.replace(to_replace=x,value=y[0]+y[1])
df2.tail(5)


# In[6]:


df2.corr(numeric_only=True)


# In[7]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[8]:


X= df2.iloc[:,1:14]
Y= df2.iloc[:,-1]   


# In[10]:


best_features= SelectKBest(score_func=chi2, k=3,)
fit= best_features.fit(X,Y)


# In[ ]:


df_scores= pd.DataFrame(fit.scores_)
df_columns= pd.DataFrame(X.columns)


# In[ ]:


features_scores= pd.concat([df_columns, df_scores], axis=1)
features_scores.columns= ['Features', 'Score']
features_scores.sort_values(by = 'Score')


# In[ ]:


X= df2[['SEP', 'JUN', 'JUL']]  the top 3 features
Y= df2[['Sem_Pen_Amt']]  the target output


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=100)


# In[ ]:


logreg= LogisticRegression()
logreg.fit(X_train,y_train)


# In[ ]:


y_pred=logreg.predict(X_test)
print (X_test) 
print (y_pred)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import classification_report


# In[ ]:


print(‘Accuracy: ‘,metrics.accuracy_score(y_test, y_pred))
print(‘Recall: ‘,metrics.recall_score(y_test, y_pred, zero_division=1))
print(“Precision:”,metrics.precision_score(y_test, y_pred, zero_division=1))
print(“CL Report:”,metrics.classification_report(y_test, y_pred, zero_division=1))


# In[ ]:


y_pred_proba= logreg.predict_proba(X_test) [::,1]


# In[ ]:


false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba)


# In[ ]:


auc= metrics.roc_auc_score(y_test, y_pred_proba)


# In[ ]:


plt.plot(false_positive_rate, true_positive_rate)
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('false Positive Rate')
plt.legend(loc=4)


# In[ ]:




