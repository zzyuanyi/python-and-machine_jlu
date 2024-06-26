import numpy as np  
import pandas as pd
from sklearn import datasets  
from sklearn.model_selection import cross_val_score  
from sklearn.svm import SVC  
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier  
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef,confusion_matrix
data=pd.read_csv('ALL3.csv',sep='\t',index_col=0)
data=data.astype(float)
ans=[]
columns=data.columns
pos_colnums=[col for col in columns if 'POS' in col]
neg_colnums=[col for col in columns if 'NEG' in col]
num_colnum=[col for col in columns if ('POS'in col or 'NEG'in col)]
#print(num_colnum)
#print(data.iloc[0:2])
for i in data.index:
    p_data=data.loc[i,pos_colnums].values
    n_data=data.loc[i,neg_colnums].values
    x_data=data.loc[i,num_colnum].values
    t,p=stats.ttest_ind(p_data,n_data)
    #print(p_data)
    ans.append((i,t,p,p_data,n_data))
output=pd.DataFrame(ans,columns=['feature','t','p','p_data','n_data'])
#print((np.concatenate([[output.iloc[0,3]],[output.iloc[0,4]]],axis=1)))

sorted_df = output.sort_values(by='p')
tp1_data=sorted_df.head(1)
tp10_data=sorted_df.head(10)
tp100_data=sorted_df.head(100)
bt100_data=sorted_df.tail(100)
#print(bt100_data)
# 初始化模型  

models = {  
    'SVM': SVC(),  
    'Nbayes': GaussianNB(),  
    'KNN': KNeighborsClassifier()  
}  
def train_and_evaluate(data_value, labels):

    
    train_data, test_data, train_label, test_label = train_test_split(data_value, labels, test_size=0.2, random_state=30627)
    #classifiers = {'SVM': SVC(), 'Nbayes': GaussianNB(), 'KNN': KNeighborsClassifier()}
    results = {}

    for name, md in models.items():
        md.fit(train_data, train_label)
        pred = md.predict(test_data)
        acc = accuracy_score(test_label, pred)
        mcc = matthews_corrcoef(test_label, pred)
        auc=roc_auc_score(test_label,pred)
        cm = confusion_matrix(test_label, pred)
        sn = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        results[name] = [sn, sp, acc, auc, mcc]
    
    return results
feature_sets={
    "Top1":tp1_data,
    "Top10":tp10_data,
    "Top100":tp100_data,
    "Bottom100":bt100_data
}
result_dict={}
for feature_label,feature_data in feature_sets.items():
    data_value=np.concatenate([[feature_data.iloc[0,3]],[feature_data.iloc[0,4]]],axis=1).T
    data_label=np.concatenate([np.ones(feature_data.iloc[0,3].shape[0]),np.zeros(feature_data.iloc[0,4].shape[0])])
    result=train_and_evaluate(data_value,data_label)
    result_dict[feature_label]=result


norm=['Sn','Sp','Acc','Auc','Mcc']
classifier_name=['SVM','Nbayes','KNN']
norm_size=len(norm)
fig,axs=plt.subplots(2,2,figsize=(18,12))
axs=axs.flatten()
for ax,(feature_No,results) in zip(axs,result_dict.items()):
    bet=np.arange(norm_size)
    width=0.2
    for i,cn in enumerate(classifier_name):
        norm_value=results[cn]
        ax.bar(bet+(i-1)*width,norm_value,width,label=cn)
    ax.set_title(feature_No)
    ax.set_xticks(bet)
    ax.set_xticklabels(norm)
    ax.axhline(0,color='black',linewidth=0.8)
    ax.legend()
plt.tight_layout()
plt.show()


