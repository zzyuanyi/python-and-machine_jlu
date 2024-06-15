import pandas as pd
from scipy.stats import ttest_ind;
data=pd.read_csv('ALL3.csv',sep='\t',index_col=0)
data=data.astype(float)
ans=[]
columns=data.columns
pos_colnums=[col for col in columns if 'POS' in col]
neg_colnums=[col for col in columns if 'NEG' in col]
for i in data.index:
    p_data=data.loc[i,pos_colnums]
    n_data=data.loc[i,neg_colnums]
    t,p=ttest_ind(p_data,n_data)
    ans.append((i,t,p))
output=pd.DataFrame(ans,columns=['feature','t','p'])
sorted_df = output.sort_values(by='t',ascending=False)
print(sorted_df[0:10])