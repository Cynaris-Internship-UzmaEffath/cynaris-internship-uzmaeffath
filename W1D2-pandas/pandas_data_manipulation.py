import pandas as pd
df=pd.read_csv("W1D2-pandas/data/population.csv")
print(df.head(10))
print("shape:",df.shape)
print("Datatypes:", df.dtypes)
df1=df[df["State"]>"50000000"]
print(df1)
df2=df.groupby("Capital")["State"].count()
print(df2)
df3=df[["State","Capital"]]
merge_df=df.merge(df3,on="State")
print(merge_df.head())
pivot_df=pd.pivot_table(df,values="Literacy Rate (%)", index="Rank")
print("pivot_table:", pivot_df.head())
df1.to_csv("W1D2-pandas/output/cleaned_data.csv", index=False)
df1.to_parquet("W1D2-pandas/output/cleaned_data.parquet", index=False)
