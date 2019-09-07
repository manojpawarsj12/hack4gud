import pandas as pd
data = {'datetime':["a","b","c","d"],'attendence':[0,0,0,0],
'regno':[38110078,38110271,38110306,38110301,]
}
df=pd.DataFrame(data=data,index=['Bharat', 'Kriti', 'dikshita', 'manju']) 
print(df.head())
df.loc['Bharat','attendence']=1
print(df.head())