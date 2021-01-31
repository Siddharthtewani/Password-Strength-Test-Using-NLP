#%%
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#%%
# df=pd.read_csv(r"C:\Users\siddh\Desktop\Projects\Paswword_Strong_Or_Not_NLP/Data.csv",error_bad_lines=False)
df=pd.read_csv(r"C:\Users\siddh\Desktop\Projects\Paswword_Strong_Or_Not_NLP/Data.csv",error_bad_lines=False)
df.head()
# %%
df.isnull().sum()
# %%
df.shape
# %%
df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4","Unnamed: 5","Unnamed: 6"],axis=1,inplace=True)
# %%

df.dropna(inplace=True)
df.isnull().sum()
# %%
df["strength"].value_counts()
# %%
data=df[df["strength"]=="0"]
# %%
df1=df[df["strength"]=="1"]
# %%
df2=df[df["strength"]=="2"]
# %%
df=pd.concat([data,df1,df2],axis=0)
# %%
df.shape
# %%
df["strength"].value_counts()
sns.countplot(df["strength"])
# %%
password_tuple=np.array(df)
password_tuple
# %%
import random
random.shuffle(password_tuple)
# %%
x=[label[0] for label in password_tuple]
y=[label[1] for label in password_tuple]
x
# %%
y
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
