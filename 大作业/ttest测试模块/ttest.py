import pandas as pd
from scipy.stats import ttest_ind;
data=pd.read_csv('ttest_data.csv')
data.set_index(data.columns[0], inplace=True)
data.index.name = 'diagnosis'
data=data.astype(float)
#only_data=data.drop(columns=['id','Unnamed: 32'])
#data_T=(only_data.T)
#data_T.to_csv('ttest_data.csv', index=True, header=False)
#print(data)

ans=[]
columns=data.columns
m_columns=[col for col in columns if 'M' in col]
b_columns=[col for col in columns if 'B' in col]
#print(pos_columns)
for i in data.index:
    m_data=data.loc[i,m_columns]
    b_data=data.loc[i,b_columns]
    t,p=ttest_ind(m_data,b_data)
    ans.append((i,t,p))
output=pd.DataFrame(ans,columns=['feature','t','p'])
#print(output)
sorted_df_t = output.sort_values(by='t',ascending=False)
#print(sorted_df_t[0:30])
sorted_df_p = output.sort_values(by='p',ascending=True)
print(sorted_df_p[0:30])