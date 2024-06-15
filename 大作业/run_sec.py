import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFE 
import matplotlib.pyplot as plt
import numpy as np
import random 
import seaborn as sns 
from sklearn.feature_selection import VarianceThreshold  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data=pd.read_csv('data.csv')
label_raw=data.diagnosis
only_data=data.drop(columns=['id','diagnosis','Unnamed: 32'])
sel=VarianceThreshold(threshold=(0.8)*0.2)
data_sel=sel.fit_transform(only_data)
#print(y_train_sel)

#print(only_data)
#print(data_sel)
'''
X_train, X_test, y_train, y_test = train_test_split(only_data, label_raw, test_size=0.2, random_state=42)
clf=KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
sn = cm[1, 1] / (cm[1, 1] + cm[1, 0])
sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
#results[name] = [sn, sp, acc, mcc]
sns.heatmap(cm,annot=True,fmt='d')
plt.show()
'''

def train_and_evaluate(X, labels):
    if X is None or labels is None:
        print("Error: Invalid feature set or labels.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    classifiers = {'SVM': SVC(), 'Nbayes': GaussianNB(), 'KNN': KNeighborsClassifier(),'RandomForest':RandomForestClassifier(100,random_state=42)}
    results = {}
    fig, axs = plt.subplots(2, 2)
    i=0
    for name, clf in classifiers.items():
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        sn = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        results[name] = [sn, sp, acc, mcc]
        sns.heatmap(cm,ax=axs[i//2,i%2],annot=True,fmt='.1f')
        i=i+1
        #plt.show()
        print(acc)
    #plot_histograms(results)
    plt.show()
    
    return results
def sel_train_and_evaluate(X, labels):
    if X is None or labels is None:
        print("Error: Invalid feature set or labels.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    classifiers = {'SVM': SVC(), 'Nbayes': GaussianNB(), 'KNN': KNeighborsClassifier(),'RandomForest':RandomForestClassifier()}
    results = {}
    fig, axs = plt.subplots(2, 2)
    i=0
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        sn = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        results[name] = [sn, sp, acc, mcc]
        sns.heatmap(cm,ax=axs[i//2,i%2],annot=True,fmt='f')
        i=i+1
        print(acc)
        #plt.show()
    plt.show() 
    return results

#data_vol=only_data.drop(['texture_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','factal_dimesion_mean',''])
data_vol = only_data.filter(regex='^(radius_mean|perimeter_mean|concave points_mean|radius_worst|perimeter_worst|area_worst)')
#print(data_vol)
def vol_train_and_evaluate(X, labels):
    if X is None or labels is None:
        print("Error: Invalid feature set or labels.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    classifiers = {'SVM': SVC(), 'Nbayes': GaussianNB(), 'KNN': KNeighborsClassifier(),'RandomForest':RandomForestClassifier()}
    results = {}
    fig, axs = plt.subplots(2, 2)
    i=0
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        sn = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        results[name] = [sn, sp, acc, mcc]
        print(acc)
        sns.heatmap(cm,ax=axs[i//2,i%2],annot=True,fmt='.1f')
        i=i+1
        #plt.show()
    plt.show() 
    return results

def sec1_train_and_evaluate(X, labels):
    if X is None or labels is None:
        print("Error: Invalid feature set or labels.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    select_feature = SelectKBest(chi2, k=6).fit(X_train, y_train)
#print(select_feature.get_support())
#print(only_data)
    X_train_fe=select_feature.transform(X_train)
    X_test_fe=select_feature.transform(X_test)
    classifiers = {'SVM': SVC(), 'Nbayes': GaussianNB(), 'KNN': KNeighborsClassifier(),'RandomForest':RandomForestClassifier()}
    results = {}
    fig, axs = plt.subplots(2, 2)
    i=0
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        sn = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        results[name] = [sn, sp, acc, mcc]
        print(acc)
        sns.heatmap(cm,ax=axs[i//2,i%2],annot=True,fmt='.1f')
        i=i+1
        #plt.show()
    plt.show() 
    return results

def sec2_train_and_evaluate(X, labels):
    if X is None or labels is None:
        print("Error: Invalid feature set or labels.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    #select_feature = SelectKBest(chi2, k=6).fit(X_train, y_train)
#print(select_feature.get_support())
#print(only_data)
    #X_train_fe=select_feature.transform(X_train)
    #X_test_fe=select_feature.transform(X_test)
    classifiers = { 'RandomForest':RandomForestClassifier()}
    results = {}
    fig, axs = plt.subplots(2, 2)
    i=0
    for name, clf in classifiers.items():
        rfe = RFE(estimator=clf, n_features_to_select=6, step=1)
        rfe.fit(X_train,y_train)
        X_train=rfe.transform(X_train)
        X_test=rfe.transform(X_test)
        clf.fit(X_train, y_train)
        #print(rfe.get_support())
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        sn = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        results[name] = [sn, sp, acc, mcc]
        print(acc)
        sns.heatmap(cm,ax=axs[i//2,i%2],annot=True,fmt='.1f')
        i=i+1
        #plt.show()
    plt.show() 
    return results
train_and_evaluate(only_data,label_raw)   
#sel_train_and_evaluate(data_sel,label_raw)    
#vol_train_and_evaluate(data_vol,label_raw)    
#sec1_train_and_evaluate(only_data,label_raw)
#sec2_train_and_evaluate(only_data,label_raw)