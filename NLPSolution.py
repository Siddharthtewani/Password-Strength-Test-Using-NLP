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
# %%git 
x=[label[0] for label in password_tuple]
y=[label[1] for label in password_tuple]
# %%
def characters(input):
    charac=[]
    for i in input:
        charac.append(i)
    return charac
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer(tokenizer=characters)
x=vector.fit_transform(x)
# %%
x.shape
# %%
first=x[0]
first.T.todense()
# %%
df=pd.DataFrame(first.T.todense(),index=vector.get_feature_names(),columns=["TF-IDF"])
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
# %%
X_train.shape
# %%
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,multi_class='multinomial')
# %%
clf.fit(X_train,y_train)
# %%
dt=np.array(['ItsMeSiddharth2000#%!!'])
pred=vector.transform(dt)
clf.predict(pred)
# %%
y_pred=clf.predict(X_test)
y_pred
# %%
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
# %%
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

