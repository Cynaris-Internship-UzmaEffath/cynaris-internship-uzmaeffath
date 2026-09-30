import pandas as pd
import numpy as np
df=pd.read_csv("W1D3-data-loading-cleaning/data/raw_data.csv")
print("original data:")
print(df.head())
print("\nrows with no missing values:")
df1=df.dropna(axis=0,how="any")
print(df1)
df['J']=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
print("\data after adding an empty column:")
df2=df.dropna(axis=1,how="all")
print("\nafter removing completely empty columns:")
print(df2)
df3=df.dropna(thresh=5,axis=1)
print("\ncouluns with atleast 5 non-empty values:")
print(df3)
df4=df.dropna(subset=['Name','City'])
print("\nrows where name and city are available:")
print(df4)
df5=df.fillna(0)
print("\nmissing values replaced with 0:")
print(df5)
df6=df['Name'].ffill()
print("\nforward fill values:")
print(df6)
df7=df['Department'].bfill()
print("\nbackward  fill values:")
print(df7)
df8=df.fillna({'Name':'Taehyung','City':'Korea'})
print("\nfill with d/f values per column:")
print(df8)
df9=df.drop_duplicates()
print("\ndrop duplicates:")
print(df9)
df10=df.drop_duplicates(subset=['Age'])
print("\nonly check Coumn  age:",df10)
df11=df.drop_duplicates(subset=['Age','Salary'],keep='last')
print("\n check age and salary:", df11)
df12=df.replace({'Name':{'Rahul':'RAHUL'}})
print("\nrepalce values:",df12)
df['Age']=df['Age'].astype('Float64')
print("\ntype conversion:",df)
df['Name']=df['Name'].str.strip()
print("\n remove spaces:",df)
df['Department'] = df['Department'].map({
    'IT': 1,
    'HR': 2,
    'Finance': 3,
    'Sales': 4
})
print("\nmapping and replacing values:",df)
# 8. Handling outliers

lower = df['Age'].quantile(0.05)
upper = df['Age'].quantile(0.95)

df['Age'] = df['Age'].clip(lower=lower, upper=upper)

print("\nAge after handling outliers:")
print(df['Age'])
# 9. Apply a function using lambda

df['Name'] = df['Name'].apply(
    lambda x: x.strip().lower() if isinstance(x, str) else x
)

print(df)