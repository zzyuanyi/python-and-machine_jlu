import pandas as pd
from scipy.stats import ttest_ind;
import matplotlib.pyplot as plt  
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
    p_data=data.loc[i,pos_colnums]
    n_data=data.loc[i,neg_colnums]
    x_data=data.loc[i,num_colnum]
    t,p=ttest_ind(p_data,n_data)
    #print(p_data)
    ans.append((i,t,p,p_data,n_data))
output=pd.DataFrame(ans,columns=['feature','t','p','p_data','n_data'])
sorted_df = output.sort_values(by='t',ascending=False)
#alpha = 0.05  
#print(sorted_df.iloc[0])

#print(group1)
#print(group2)
group1=sorted_df.iloc[0,3]
group2=sorted_df.iloc[1,3]
group3=sorted_df.iloc[0,4]
group4=sorted_df.iloc[1,4]
plt.subplot(2, 2, 1)
plt.scatter( group1,group2, label='POS', color='blue') 
plt.scatter(group3,group4, label='NEG', color='red')
plt.legend()  
plt.xlabel('Rank1-value')  
plt.ylabel('Rank2-value')  
plt.title('Rank1 vs Rank2 based t-test')
group1=sorted_df.iloc[8,3]
group2=sorted_df.iloc[9,3]
group3=sorted_df.iloc[8,4]
group4=sorted_df.iloc[9,4]
plt.subplot(2, 2, 2)
plt.scatter( group1,group2, label='POS', color='blue') 
plt.scatter(group3,group4, label='NEG', color='red')
plt.legend()  
plt.xlabel('Rank9-value')  
plt.ylabel('Rank10-value')  
plt.title('Rank9 vs Rank10 based t-test')
group1=sorted_df.iloc[999,3]
group2=sorted_df.iloc[1000,3]
group3=sorted_df.iloc[999,4]
group4=sorted_df.iloc[1000,4]
plt.subplot(2, 2, 3)
plt.scatter( group1,group2, label='POS', color='blue') 
plt.scatter(group3,group4, label='NEG', color='red')
plt.legend()  
plt.xlabel('Rank1000-value')  
plt.ylabel('Rank1001-value')  
plt.title('Rank1000 vs Rank1001 based t-test') 
group1=sorted_df.iloc[9999,3]
group2=sorted_df.iloc[10000,3]
group3=sorted_df.iloc[9999,4]
group4=sorted_df.iloc[10000,4]
plt.subplot(2, 2, 4)
plt.scatter( group1,group2, label='POS', color='blue') 
plt.scatter(group3,group4, label='NEG', color='red')
plt.legend()  
plt.xlabel('Rank10000-value')  
plt.ylabel('Rank10001-value')  
plt.title('Rank10000 vs Rank10001 based t-test')
plt.gcf().set_size_inches(12, 10) 
plt.show()  

