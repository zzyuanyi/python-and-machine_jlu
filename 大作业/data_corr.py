import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import random 
import seaborn as sns 
data=pd.read_csv('data.csv')
label_raw=data.diagnosis
only_data=data.drop(columns=['id','diagnosis','Unnamed: 32'])
#print(only_data)
ax=sns.countplot(x='diagnosis',data=data,label="Number")
#plt.show()
#print(only_data.shape)
data_std=((only_data-only_data.mean())/only_data.std())
data_v1=pd.concat([label_raw,data_std],axis=1)
data_v1=pd.melt(data_v1,id_vars='diagnosis',var_name='features')

sns.violinplot(x='features',y='value',hue='diagnosis',data=data_v1,split=True,inner='quart')
plt.xticks(rotation=90)
#plt.show()
#sns.jointplot(x='A',y='B',data=only_data, kind="reg", color="#ce1414")/////暂时不确定绘图的方式
plt.figure(figsize=(8,6))
sns.heatmap(only_data.corr(),linewidth=0.5,cmap='viridis')
plt.show()
#print(label_raw)